# Copyright (c) ModelScope Contributors. All rights reserved.
# Label: 80327919
# Date: 2026-07-14 18:33:42
"""
CrossTokenLoss: Projection-Guided Cross-Tokenizer Knowledge Distillation Loss.

This module implements the X-Token approach with enhanced H-KL loss that includes
ULD (Unmatched Logit Distillation) for student tokens that cannot be mapped to
any teacher token via the projection matrix.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)

Key differences from CTKDLoss:
    1. H-KL loss now includes ULD (Unmatched Logit Distillation) component:
       Loss = gamma_kl * common-KL + gamma_uld * L_uld
    2. ULD handles student tokens with no projection mapping by computing
       standard KL divergence in the student vocabulary space.
    3. Cleaner code structure with reduced debug noise.
"""
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import hashlib
import pickle
import threading

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Global cache for projection matrices to avoid recomputation
_CROSS_TOKEN_PROJECTION_CACHE = {}


class CrossTokenLoss(Loss):
    """Cross-Tokenizer Knowledge Distillation Loss with ULD support.

    This loss extends the X-Token approach by adding ULD (Unmatched Logit
    Distillation) to the H-KL loss type. For student tokens that cannot be
    mapped to any teacher token via the projection matrix W, ULD computes
    a standard KL divergence in the student vocabulary space.

    Projection Matrix Construction:
        1. Initialize W[s,t] = 0 for all student token s and teacher token t
        2. Exact match: If student token text equals teacher token text, W[s,t] = 1
        3. Multi-token decoding: For unmatched student tokens, decode to text and
           re-encode with teacher tokenizer. If sequence length < L (max_length),
           assign weights: W[s, t[i]] = beta * gamma^i

    Loss Types:
        - P-KL: Projects student distribution to teacher vocabulary space using
          the full projection matrix W, then computes KL divergence.
        - H-KL: Uses best teacher token mapping (argmax W[s,t]) for common-KL,
          plus ULD for unmatched student tokens.
          Total: Loss = gamma_kl * common-KL + gamma_uld * L_uld

        vocab_topk: Top-k size for KL computation (None = use full vocabulary).
            When set, projects student to teacher vocab space, then selects the
            top-k most important teacher vocab indices (per-batch, per-vocab-max),
            slices both distributions to the same k columns, renormalizes,
            and computes KL. This follows the NeMo-RL P-KL approach:
            "project → teacher-side top-k → dual-slice to same k → renormalize → KL".
            Reduces memory and computation while maintaining semantic consistency.

    Args:
        student_tokenizer: Tokenizer for the student model.
        teacher_tokenizer_group: List of tokenizers for the teacher models.
        teacher_weights: Optional list of weights for each teacher (default: equal weights).
        max_length: Maximum span length L for multi-token matching (default: 4).
        beta: Base weight beta for projection (default: 0.9).
        gamma: Decay rate gamma for multi-token weights (default: 0.1).
        loss_type: Type of KL loss - 'pkl' for P-KL or 'hkl' for H-KL (default: auto).
        temperature: Temperature for softmax in KL divergence (default: 1.0).
        gamma_kl: Weight for common-KL in H-KL loss (default: 1.0).
        gamma_uld: Weight for ULD loss in H-KL loss (default: 0.5).
        device: Device to place the projection matrices on (default: auto).

    Example:
        >>> from transformers import AutoTokenizer
        >>> student_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        >>> teacher_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> loss_fn = CrossTokenLoss(student_tok, [teacher_tok], loss_type='hkl')
    """

    def __init__(
        self,
        student_tokenizer: 'PreTrainedTokenizer',
        teacher_tokenizer_group: list,
        teacher_weights: Optional[list] = None,
        max_length: int = 4,
        beta: float = 0.9,
        gamma: float = 0.1,
        loss_type: Optional[str] = None,
        temperature: float = 1.0,
        gamma_kl: float = 1.0,
        gamma_uld: float = 0.5,
        vocab_topk: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer_group = teacher_tokenizer_group
        self.num_teachers = len(teacher_tokenizer_group)

        # Set teacher weights (default to equal weights)
        if teacher_weights is None:
            self.teacher_weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            if len(teacher_weights) != self.num_teachers:
                raise ValueError(
                    f"Number of weights ({len(teacher_weights)}) must match "
                    f"number of teachers ({self.num_teachers})"
                )
            weight_sum = sum(teacher_weights)
            self.teacher_weights = [w / weight_sum for w in teacher_weights]

        self.max_length = max_length
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.gamma_kl = gamma_kl
        self.gamma_uld = gamma_uld
        self.vocab_topk = vocab_topk if vocab_topk is not None else 256

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = torch.device('npu:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Auto-select loss_type based on vocabulary coverage
        if loss_type is None:
            self.loss_type = self._auto_select_loss_type()
        else:
            self.loss_type = loss_type

        # Vocabulary sizes
        self.student_vocab_size = len(student_tokenizer)
        self.teacher_vocab_sizes = [len(tok) for tok in teacher_tokenizer_group]

        # Lazy initialization flags
        self._projection_matrices_built = False
        self.projection_matrices: list = []
        self.projection_student_indices_list: list = []
        self.projection_teacher_indices_list: list = []
        self.projection_values_list: list = []
        self._best_teacher_mappings: list = []
        self._unmatched_student_masks: list = []  # For ULD: boolean mask per teacher
        self._build_lock = threading.Lock()

    # =========================================================================
    # Loss type selection
    # =========================================================================

    def _auto_select_loss_type(self) -> str:
        """Auto-select loss_type based on vocabulary coverage.

        High coverage (>= 0.7) -> 'hkl', low coverage -> 'pkl'.
        """
        coverages = [
            self._calculate_vocab_coverage(self.student_tokenizer, tok)
            for tok in self.teacher_tokenizer_group
        ]
        avg_coverage = sum(coverages) / len(coverages)
        return 'hkl' if avg_coverage >= 0.7 else 'pkl'

    @staticmethod
    def _calculate_vocab_coverage(student_tokenizer, teacher_tokenizer) -> float:
        """Calculate vocabulary coverage = |intersection| / |union|."""
        student_vocab = set(student_tokenizer.get_vocab().keys())
        teacher_vocab = set(teacher_tokenizer.get_vocab().keys())
        intersection = student_vocab & teacher_vocab
        union = student_vocab | teacher_vocab
        return len(intersection) / len(union) if len(union) > 0 else 0.0

    # =========================================================================
    # Projection matrix construction
    # =========================================================================

    def _ensure_projection_matrices_built(self):
        """Ensure projection matrices are built (lazy initialization with caching)."""
        if self._projection_matrices_built:
            return

        with self._build_lock:
            if self._projection_matrices_built:
                return

            cache_key = self._generate_cache_key()

            if cache_key in _CROSS_TOKEN_PROJECTION_CACHE:
                cached = _CROSS_TOKEN_PROJECTION_CACHE[cache_key]
                self.projection_matrices = cached['projection_matrices']
                self.projection_student_indices_list = [
                    t.to(self.device) for t in cached['projection_student_indices_list']
                ]
                self.projection_teacher_indices_list = [
                    t.to(self.device) for t in cached['projection_teacher_indices_list']
                ]
                self.projection_values_list = [
                    t.to(self.device) for t in cached['projection_values_list']
                ]
            else:
                self.projection_matrices = []
                self.projection_student_indices_list = []
                self.projection_teacher_indices_list = []
                self.projection_values_list = []
                for i, teacher_tok in enumerate(self.teacher_tokenizer_group):
                    self._build_projection_matrix_for_teacher(teacher_tok, i)

                _CROSS_TOKEN_PROJECTION_CACHE[cache_key] = {
                    'projection_matrices': self.projection_matrices,
                    'projection_student_indices_list': self.projection_student_indices_list,
                    'projection_teacher_indices_list': self.projection_teacher_indices_list,
                    'projection_values_list': self.projection_values_list,
                }

            # For H-KL: precompute best teacher mapping and unmatched masks
            if self.loss_type == 'hkl':
                self._best_teacher_mappings = []
                self._unmatched_student_masks = []
                for i in range(self.num_teachers):
                    self._build_best_teacher_mapping_for_teacher(i)

            self._projection_matrices_built = True

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on tokenizer configurations."""
        config_data = {
            'student_vocab': self.student_tokenizer.get_vocab(),
            'teacher_vocabs': [tok.get_vocab() for tok in self.teacher_tokenizer_group],
            'max_length': self.max_length,
            'beta': self.beta,
            'gamma': self.gamma,
        }
        return hashlib.md5(pickle.dumps(config_data)).hexdigest()

    def _build_projection_matrix_for_teacher(self, teacher_tokenizer, teacher_index: int):
        """Build the sparse projection matrix W for a specific teacher tokenizer.

        Steps:
            1. Initialize W[s,t] = 0
            2. Exact match: W[s,t] = 1 if decoded text matches
            3. Multi-token decoding: W[s, t[i]] = beta * gamma^i for unmatched tokens
        """
        teacher_vocab_size = len(teacher_tokenizer)

        student_indices = []
        teacher_indices = []
        values = []

        # Step 2: Exact match - build text-to-id mapping for teacher tokens
        teacher_token_text_to_id = {}
        for token_id in range(teacher_vocab_size):
            token_text = teacher_tokenizer.decode([token_id], skip_special_tokens=False).strip()
            teacher_token_text_to_id[token_text] = token_id

        matched_student_ids = set()
        for student_id in range(self.student_vocab_size):
            student_token_text = self.student_tokenizer.decode(
                [student_id], skip_special_tokens=False
            ).strip()
            if student_token_text in teacher_token_text_to_id:
                teacher_id = teacher_token_text_to_id[student_token_text]
                student_indices.append(student_id)
                teacher_indices.append(teacher_id)
                values.append(1.0)
                matched_student_ids.add(student_id)

        # Step 3: Multi-token decoding match for unmatched student tokens
        for student_id in range(self.student_vocab_size):
            if student_id in matched_student_ids:
                continue

            text = self.student_tokenizer.decode([student_id], skip_special_tokens=False)
            if not text or not text.strip():
                continue

            teacher_token_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
            seq_length = len(teacher_token_ids)

            if 0 < seq_length < self.max_length:
                for i, teacher_token_id in enumerate(teacher_token_ids):
                    weight = self.beta * (self.gamma ** i)
                    student_indices.append(student_id)
                    teacher_indices.append(teacher_token_id)
                    values.append(weight)

        # Convert to tensors
        student_indices_tensor = torch.tensor(student_indices, dtype=torch.long, device=self.device)
        teacher_indices_tensor = torch.tensor(teacher_indices, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.projection_student_indices_list.append(student_indices_tensor)
        self.projection_teacher_indices_list.append(teacher_indices_tensor)
        self.projection_values_list.append(values_tensor)
        self.projection_matrices.append(None)  # Dense matrix not stored to save memory

        nnz = len(student_indices)
        print(f"[CrossToken] Built projection matrix for teacher {teacher_index}: "
              f"{nnz} mappings, student_vocab={self.student_vocab_size}, "
              f"teacher_vocab={teacher_vocab_size}")

    def _build_best_teacher_mapping_for_teacher(self, teacher_index: int):
        """Build best teacher token mapping for H-KL and unmatched student mask for ULD.

        For each student token:
            - Find the teacher token with highest projection weight (for common-KL)
            - Mark whether it has any mapping at all (for ULD)
        """
        student_indices = self.projection_student_indices_list[teacher_index]
        teacher_indices = self.projection_teacher_indices_list[teacher_index]
        values = self.projection_values_list[teacher_index].float()

        # Initialize mapping with -1 (no mapping)
        best_teacher_mapping = torch.full(
            (self.student_vocab_size,), -1, dtype=torch.long, device=self.device
        )
        max_weights = torch.zeros(self.student_vocab_size, dtype=torch.float32, device=self.device)

        # Track which student tokens have at least one mapping
        has_mapping = torch.zeros(self.student_vocab_size, dtype=torch.bool, device=self.device)

        # Process in chunks
        chunk_size = 100000
        for i in range(0, len(student_indices), chunk_size):
            chunk_student = student_indices[i:i + chunk_size]
            chunk_teacher = teacher_indices[i:i + chunk_size]
            chunk_values = values[i:i + chunk_size]

            for j in range(len(chunk_student)):
                s_id = chunk_student[j].item()
                t_id = chunk_teacher[j].item()
                w = chunk_values[j].item()

                has_mapping[s_id] = True
                if w > max_weights[s_id].item():
                    max_weights[s_id] = w
                    best_teacher_mapping[s_id] = t_id

        # Unmatched mask: True for student tokens with NO mapping (for ULD)
        unmatched_mask = ~has_mapping

        self._best_teacher_mappings.append(best_teacher_mapping)
        self._unmatched_student_masks.append(unmatched_mask)

        matched_count = has_mapping.sum().item()
        unmatched_count = unmatched_mask.sum().item()
        print(f"[CrossToken] Teacher {teacher_index} mapping: "
              f"matched={matched_count}, unmatched={unmatched_count} "
              f"(ULD will handle {unmatched_count} unmatched tokens)")

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        """Compute CrossToken loss between student and teacher models.

        Args:
            inputs: Dict containing 'input_ids' and 'labels' for student model.
            outputs: Dict containing 'logits' from student logits.
            teacher_logits_group: List of teacher model logits for each teacher.
            teacher_topk_logprobs_group: List of teacher topk logprobs for each teacher.
            teacher_topk_indices_group: List of teacher topk indices for each teacher.
            teacher_input_ids_group: List of teacher input_ids for each teacher (for text decoding).
            **kwargs: Additional arguments.

        Returns:
            LossOutput with the computed loss and number of tokens.
        """
        self._ensure_projection_matrices_built()

        student_logits = outputs.get('logits')
        if student_logits is None:
            raise ValueError("Student logits not found in outputs")

        student_labels = inputs.get('labels')
        if student_labels is None:
            raise ValueError("Student labels not found in inputs")

        # Get student input_ids for text decoding in P-KL
        student_input_ids = inputs.get('input_ids')
        if student_input_ids is None:
            student_input_ids = outputs.get('input_ids')

        # Extract teacher logits group
        teacher_logits_group = kwargs.get('teacher_logits_group')
        if teacher_logits_group is None:
            teacher_logits_group = outputs.get('teacher_logits_group')

        # Support topk format from vLLM
        teacher_topk_logprobs_group = kwargs.get('teacher_topk_logprobs_group')
        teacher_topk_indices_group = kwargs.get('teacher_topk_indices_group')

        # Extract teacher input_ids group for text decoding in P-KL
        teacher_input_ids_group = kwargs.get('teacher_input_ids_group')
        if teacher_input_ids_group is None:
            teacher_input_ids_group = outputs.get('teacher_input_ids_group')

        if teacher_logits_group is None and teacher_topk_logprobs_group is not None:
            if (len(teacher_topk_logprobs_group) != self.num_teachers or
                    len(teacher_topk_indices_group) != self.num_teachers):
                raise ValueError(
                    f"Number of teachers in topk format ({len(teacher_topk_logprobs_group)}) "
                    f"must match number of teachers ({self.num_teachers})"
                )

            teacher_logits_group = []
            for i in range(self.num_teachers):
                teacher_topk_logprobs = teacher_topk_logprobs_group[i]
                teacher_topk_indices = teacher_topk_indices_group[i]
                vocab_size = self.teacher_vocab_sizes[i]
                batch_size, seq_len, topk = teacher_topk_logprobs.shape

                # Convert logprobs to probabilities (vLLM returns log-probs, not logits)
                teacher_probs_full = torch.zeros(
                    (batch_size, seq_len, vocab_size),
                    dtype=teacher_topk_logprobs.dtype,
                    device=student_logits.device,
                )
                teacher_topk_probs = torch.exp(teacher_topk_logprobs)
                teacher_probs_full.scatter_(
                    dim=2,
                    index=teacher_topk_indices.to(teacher_probs_full.device),
                    src=teacher_topk_probs.to(teacher_probs_full.device),
                )
                teacher_logits_group.append(teacher_probs_full)

        if teacher_logits_group is None:
            raise ValueError(
                "Teacher logits group not found. Provide either teacher_logits_group "
                "or (teacher_topk_logprobs_group + teacher_topk_indices_group)"
            )

        if len(teacher_logits_group) != self.num_teachers:
            raise ValueError(
                f"Number of teacher logits ({len(teacher_logits_group)}) must match "
                f"number of teachers ({self.num_teachers})"
            )

        # Get labels - prefer teacher_labels from kwargs
        teacher_labels_group = kwargs.get('teacher_labels')
        if teacher_labels_group is not None and len(teacher_labels_group) > 0:
            labels = teacher_labels_group[0].to(student_logits.device)
        else:
            labels = inputs.get('labels')
            if labels is None:
                raise ValueError("labels not found in inputs")

        # ---- [DEBUG] 打印输入统计 ----
        print(f"\n========== CrossTokenLoss DEBUG ==========")
        print(f"student_logits shape: {student_logits.shape}")
        print(f"student_logits range: [{student_logits.min().item():.4f}, {student_logits.max().item():.4f}]")
        print(f"student_logits mean: {student_logits.mean().item():.4f}, std: {student_logits.std().item():.4f}")
        print(f"labels shape: {labels.shape}")
        print(f"labels valid tokens (non -100): {labels.ne(-100).sum().item()}")
        print(f"labels total tokens: {labels.numel()}")
        print(f"labels valid ratio: {labels.ne(-100).sum().item() / labels.numel():.4f}")
        print(f"===========================================")

        # Compute weighted loss across all teachers
        total_loss = 0.0
        teacher_losses = []
        for i in range(self.num_teachers):
            teacher_logits = teacher_logits_group[i]
            print(f"teacher_logits shape: {teacher_logits.shape}")
            weight = self.teacher_weights[i]

            # Get teacher input_ids for this teacher (if available)
            teacher_input_ids = None
            if teacher_input_ids_group is not None and i < len(teacher_input_ids_group):
                teacher_input_ids = teacher_input_ids_group[i]

            if self.loss_type == 'pkl':
                teacher_loss = self._compute_pkl_loss(
                    student_logits, teacher_logits, labels, i,
                    student_input_ids=student_input_ids,
                    teacher_input_ids=teacher_input_ids
                )
                loss_components = {'pkl_loss': teacher_loss.item()}
            elif self.loss_type == 'hkl':
                teacher_loss, common_kl, uld_loss = self._compute_hkl_loss(
                    student_logits, teacher_logits, labels, i
                )
                loss_components = {
                    'common_kl': common_kl.item(),
                    'uld_loss': uld_loss.item() if uld_loss is not None else 0.0,
                    'total_hkl': teacher_loss.item(),
                }
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'pkl' or 'hkl'")

            weighted_loss = weight * teacher_loss
            total_loss += weighted_loss
            teacher_losses.append({
                'teacher_index': i,
                'loss_type': self.loss_type,
                'weight': weight,
                **loss_components,
                'weighted_loss': weighted_loss.item(),
            })

        # Logging
        print(f"\n=== CrossTokenLoss Breakdown ===")
        print(f"Loss Type: {self.loss_type} | Teachers: {self.num_teachers}")
        for info in teacher_losses:
            print(f"  Teacher {info['teacher_index']} (w={info['weight']:.4f}): {info}")
        print(f"Total Loss: {total_loss.item():.6f}")
        print("=" * 40)

        # Calculate num_tokens
        if teacher_logits_group is not None and len(teacher_logits_group) > 0:
            num_tokens = teacher_logits_group[0].ne(-100).sum().item()
        else:
            num_tokens = labels.ne(-100).sum().item() if labels is not None else 0

        return LossOutput(loss=total_loss, num_tokens=num_tokens)

    # =========================================================================
    # P-KL loss
    # =========================================================================

    def _compute_pkl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_index: int = 0,
        student_input_ids: Optional[torch.Tensor] = None,
        teacher_input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute P-KL (Projection KL) loss following NeMo framework approach.

        This implementation follows the P-KL approach from NeMo framework:
        1. Project student probabilities to teacher vocabulary space using projection matrix
        2. Apply temperature scaling to both student and teacher distributions
        3. Compute KL divergence between projected student and teacher distributions

        Args:
            student_logits: [batch, seq_len, student_vocab_size]
            teacher_logits: [batch, seq_len, teacher_vocab_size] (raw logits, not probabilities)
            labels: [batch, seq_len]
            teacher_index: Index of the teacher model
            student_input_ids: [batch, seq_len] original student input token IDs (unused in this version)
            teacher_input_ids: [batch, seq_len] original teacher input token IDs (unused in this version)

        Returns:
            Scalar loss value.
        """
        print(f'student_logits shape: {student_logits.shape}')
        print(f'student_logits dtype: {student_logits.dtype}')
        print(f'student_logits range: [{student_logits.min().item():.4f}, {student_logits.max().item():.4f}]')
        print(f'student_logits mean: {student_logits.mean().item():.4f}, std: {student_logits.std().item():.4f}')
        print(f'student_logits device: {student_logits.device}')

        print(f'teacher_logits shape: {teacher_logits.shape}')
        print(f'teacher_logits dtype: {teacher_logits.dtype}')
        print(f'teacher_logits range: [{teacher_logits.min().item():.4f}, {teacher_logits.max().item():.4f}]')
        print(f'teacher_logits mean: {teacher_logits.mean().item():.4f}, std: {teacher_logits.std().item():.4f}')
        print(f'teacher_logits device: {teacher_logits.device}')
        # Shift for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().to(student_logits.device)
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)

        loss_mask = (shift_labels != -100).float()

        batch_size, seq_len, student_vocab_size = shift_student_logits.shape
        teacher_vocab_size = shift_teacher_logits.shape[-1]

        # Apply temperature scaling and compute probabilities
        student_probs = F.softmax(shift_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        # Get projection matrix data for this teacher
        student_indices = self.projection_student_indices_list[teacher_index].to(student_probs.device)
        teacher_indices = self.projection_teacher_indices_list[teacher_index].to(student_probs.device)
        proj_values = self.projection_values_list[teacher_index].to(student_probs.device).float()

        # Filter out-of-bounds teacher indices
        valid_mask = teacher_indices < teacher_vocab_size
        if not valid_mask.all():
            student_indices = student_indices[valid_mask]
            teacher_indices = teacher_indices[valid_mask]
            proj_values = proj_values[valid_mask]

        # Project student probabilities to teacher vocabulary space
        # Initialize projected student probabilities tensor
        projected_student_probs = torch.zeros(
            batch_size, seq_len, teacher_vocab_size,
            device=student_probs.device, dtype=student_probs.dtype
        )

        # Use scatter_add to efficiently project probabilities
        # Convert sparse indices to batch format
        batch_indices = torch.arange(batch_size, device=student_probs.device).view(-1, 1).repeat(1, seq_len).view(-1)
        seq_indices = torch.arange(seq_len, device=student_probs.device).repeat(batch_size)

        # Expand projection indices for batch processing
        expanded_student_indices = student_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        expanded_teacher_indices = teacher_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        expanded_proj_values = proj_values.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Gather student probabilities for projection
        student_probs_for_projection = student_probs.gather(
            dim=-1, index=expanded_student_indices
        )

        # Apply projection weights and scatter to teacher vocabulary
        projected_values = student_probs_for_projection * expanded_proj_values

        # Scatter the projected values to the teacher vocabulary positions
        projected_student_probs.scatter_add_(
            dim=-1,
            index=expanded_teacher_indices,
            src=projected_values
        )

        # Normalize the projected distribution to ensure it's a valid probability distribution.
        # Use direct L1 normalization instead of softmax(log(x+1e-8)) to avoid the 1e-8 "floor"
        # artificially inflating probability mass on zero entries in the large (151K) teacher vocab.
        # softmax(log(x+1e-8)) gives every zero entry exp(log(1e-8)) ≈ 1e-8 contribution,
        # and with 151,936 vocab entries the cumulative floor (~0.0015) severely dilutes
        # the non-zero projected probabilities, causing inflated KL divergence.
        proj_sum = projected_student_probs.sum(dim=-1, keepdim=True)
        # If a position has zero total projection (no student tokens map to any teacher token),
        # fall back to a uniform distribution to avoid division by zero.
        zero_sum_mask = (proj_sum <= 1e-12).squeeze(-1)
        proj_sum = proj_sum.clamp(min=1e-12)
        projected_student_probs = projected_student_probs / proj_sum
        if zero_sum_mask.any():
            # For positions with no projection at all, use uniform distribution
            projected_student_probs[zero_sum_mask] = 1.0 / teacher_vocab_size

        # NeMo-RL style top-k: select important teacher vocab indices from raw logits
        # "teacher-side top-k → dual-slice to same k → renormalize → KL"
        if self.vocab_topk is not None and self.vocab_topk < teacher_vocab_size:
            topk_indices = self._select_teacher_topk_indices(
                shift_teacher_logits, self.vocab_topk
            )  # [k]
            projected_student_probs = projected_student_probs[..., topk_indices]
            teacher_probs = teacher_probs[..., topk_indices]
            # Renormalize both in the k-dimensional subspace
            projected_student_probs = projected_student_probs / projected_student_probs.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            teacher_probs = teacher_probs / teacher_probs.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            print(f"  [P-KL top-k] vocab_topk={self.vocab_topk}, "
                  f"selected top-k indices shape: {topk_indices.shape}")

        # Compute KL divergence: KL(teacher || projected_student)
        # KL(P||Q) = Σ P(x) * (log P(x) - log Q(x))
        log_teacher = torch.log(teacher_probs + 1e-8)
        log_student = torch.log(projected_student_probs + 1e-8)

        kl_components = teacher_probs * (log_teacher - log_student)

        # Apply loss mask and compute mean KL divergence
        kl_per_position = kl_components.sum(dim=-1)  # Sum over vocabulary dimension
        masked_kl = kl_per_position * loss_mask

        # Compute average KL loss over valid positions
        valid_kl_mask = masked_kl > 0
        if valid_kl_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        loss = masked_kl.sum() / (loss_mask.sum() + 1e-8)

        # Debug information
        print(f"\n===== [P-KL NeMo Style] Teacher {teacher_index} =====")
        print(f"  student_probs shape: {student_probs.shape}")
        print(f"  teacher_probs shape: {teacher_probs.shape}")
        print(f"  projected_student_probs shape: {projected_student_probs.shape}")
        print(f"  loss_mask valid tokens: {loss_mask.sum().item()}/{loss_mask.numel()}")
        print(f"  KL loss: {loss.item():.6f}")
        print(f"=========================================")

        return loss

    # =========================================================================
    # H-KL loss (with ULD)
    # =========================================================================

    def _compute_hkl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        labels: torch.Tensor,
        teacher_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute H-KL (Heuristic KL) loss with ULD.

        H-KL uses the best teacher token mapping for each student token (common-KL),
        plus ULD for student tokens that have no mapping in the projection matrix.

        Total loss: Loss = gamma_kl * common-KL + gamma_uld * L_uld

        Args:
            student_logits: [batch, seq_len, student_vocab_size]
            teacher_probs: [batch, seq_len, teacher_vocab_size] (already probabilities)
            labels: [batch, seq_len]
            teacher_index: Index of the teacher model

        Returns:
            Tuple of (total_loss, common_kl_loss, uld_loss)
        """
        # Shift for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_probs = teacher_probs[..., :-1, :].contiguous().to(student_logits.device)
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)

        loss_mask = (shift_labels != -100).float()

        # Student probabilities with temperature
        student_probs = F.softmax(shift_student_logits / self.temperature, dim=-1)

        # Align sequence lengths (student is the reference since teacher probs
        # are already aligned to student token positions via character-level
        # matching in the data preprocessing stage).
        student_seq_len = student_probs.shape[1]
        teacher_seq_len = shift_teacher_probs.shape[1]
        if student_seq_len != teacher_seq_len:
            if teacher_seq_len > student_seq_len:
                shift_teacher_probs = shift_teacher_probs[:, :student_seq_len, :]
            else:
                pad_len = student_seq_len - teacher_seq_len
                shift_teacher_probs = F.pad(shift_teacher_probs, (0, 0, 0, pad_len))
            seq_len = student_seq_len
        else:
            seq_len = student_seq_len
        loss_mask = loss_mask[:, :seq_len]

        batch_size, seq_len, student_vocab = student_probs.shape
        teacher_vocab = shift_teacher_probs.shape[-1]

        # Get best teacher mapping and unmatched mask for this teacher
        best_mapping = self._best_teacher_mappings[teacher_index].to(student_probs.device)
        unmatched_mask = self._unmatched_student_masks[teacher_index].to(student_probs.device)

        # -- Part 1: Common-KL (for matched student tokens) --
        common_kl_loss = self._compute_common_kl(
            student_probs, shift_teacher_probs, best_mapping,
            loss_mask, batch_size, seq_len, student_vocab, teacher_vocab
        )

        # -- Part 2: ULD (for unmatched student tokens) --
        uld_loss = self._compute_uld_loss(
            student_probs, shift_teacher_probs, unmatched_mask,
            loss_mask, batch_size, seq_len, student_vocab, teacher_vocab
        )

        # -- Combine: Loss = gamma_kl * common-KL + gamma_uld * L_uld --
        if uld_loss is not None:
            total_loss = self.gamma_kl * common_kl_loss + self.gamma_uld * uld_loss
        else:
            total_loss = self.gamma_kl * common_kl_loss

        return total_loss, common_kl_loss, uld_loss

    def _compute_common_kl(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        best_mapping: torch.Tensor,
        loss_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        student_vocab: int,
        teacher_vocab: int,
    ) -> torch.Tensor:
        """Compute common-KL using best teacher token mapping.

        For each student token with a valid mapping, map its probability to the
        corresponding teacher token position, then compute KL(p_T || mapped_p_S).

        Args:
            student_probs: [batch, seq_len, student_vocab]
            teacher_probs: [batch, seq_len, teacher_vocab]
            best_mapping: [student_vocab] -> best teacher token id (-1 if no mapping)
            loss_mask: [batch, seq_len]
            batch_size, seq_len, student_vocab, teacher_vocab: dimensions

        Returns:
            Scalar common-KL loss.
        """
        # Create output tensor for mapped student probabilities
        mapped_student_probs = torch.zeros(
            batch_size, seq_len, teacher_vocab,
            device=student_probs.device, dtype=student_probs.dtype,
        )

        # For each student token with valid mapping, scatter its prob to teacher position
        valid_mask = best_mapping >= 0
        valid_student_ids = torch.where(valid_mask)[0]
        valid_teacher_ids = best_mapping[valid_student_ids]

        # Process in chunks to avoid memory issues
        chunk_size = 10000
        for i in range(0, len(valid_student_ids), chunk_size):
            chunk_student = valid_student_ids[i:i + chunk_size]
            chunk_teacher = valid_teacher_ids[i:i + chunk_size]

            mapped_student_probs.scatter_add_(
                dim=2,
                index=chunk_teacher.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                    .to(mapped_student_probs.device),
                src=student_probs[:, :, chunk_student].to(mapped_student_probs.device),
            )

        # NeMo-RL style top-k: select important teacher vocab indices
        # "teacher-side top-k → dual-slice to same k → renormalize → KL"
        if self.vocab_topk is not None and self.vocab_topk < teacher_vocab:
            topk_indices = self._select_teacher_topk_indices(
                teacher_probs, self.vocab_topk
            )  # [k]
            mapped_student_probs = mapped_student_probs[..., topk_indices]
            teacher_probs = teacher_probs[..., topk_indices]
            # Renormalize both in the k-dimensional subspace
            mapped_student_probs = mapped_student_probs / mapped_student_probs.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            teacher_probs = teacher_probs / teacher_probs.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            teacher_vocab = self.vocab_topk  # update for debug printing
            print(f"  [Common-KL top-k] vocab_topk={self.vocab_topk}, "
                  f"selected top-k indices shape: {topk_indices.shape}")

        # Compute KL divergence only over teacher non-zero positions
        # KL(p_T || mapped_p_S) = sum_{t: p_T(t)>0} p_T(t) * log(p_T(t) / mapped_p_S(t))
        teacher_nonzero_mask = teacher_probs > 0
        log_mapped_student = torch.log(mapped_student_probs + 1e-8)

        # ---- [DEBUG] Common-KL 散度计算前统计 ----
        print(f"\n  [Common-KL DEBUG]")
        print(f"    teacher_probs shape: {teacher_probs.shape}")
        print(f"    teacher_probs range: [{teacher_probs.min().item():.6f}, {teacher_probs.max().item():.6f}]")
        print(f"    teacher_probs sum per position: mean={teacher_probs.sum(dim=-1).mean().item():.6f}")
        print(f"    teacher_nonzero_ratio: {teacher_nonzero_mask.float().mean().item():.4f}")
        print(f"    mapped_student_probs range: [{mapped_student_probs.min().item():.6f}, {mapped_student_probs.max().item():.6f}]")
        print(f"    mapped_student_probs sum per position: mean={mapped_student_probs.sum(dim=-1).mean().item():.6f}")
        print(f"    mapped_student_probs == 0 ratio: {(mapped_student_probs == 0).float().mean().item():.4f}")
        print(f"    valid student->teacher mappings: {len(valid_student_ids)}/{student_vocab}")

        log_teacher = torch.log(teacher_probs + 1e-8)
        print(f"    log(teacher + 1e-8) range: [{log_teacher.min().item():.4f}, {log_teacher.max().item():.4f}]")
        print(f"    log(mapped_student + 1e-8) range: [{log_mapped_student.min().item():.4f}, {log_mapped_student.max().item():.4f}]")

        kl_contrib = teacher_probs * (
            torch.log(teacher_probs + 1e-8) - log_mapped_student
        )

        # ---- [DEBUG] KL贡献值统计 ----
        print(f"    kl_contrib (before mask) range: [{kl_contrib.min().item():.6f}, {kl_contrib.max().item():.6f}]")
        print(f"    kl_contrib (before mask) mean: {kl_contrib.mean().item():.6f}")
        print(f"    kl_contrib > 10 ratio: {(kl_contrib > 10).float().mean().item():.6f}")
        print(f"    kl_contrib > 100 ratio: {(kl_contrib > 100).float().mean().item():.6f}")
        print(f"    kl_contrib is NaN: {torch.isnan(kl_contrib).any().item()}")
        print(f"    kl_contrib is Inf: {torch.isinf(kl_contrib).any().item()}")

        kl_contrib = kl_contrib * teacher_nonzero_mask
        kl_div = kl_contrib.sum(dim=-1)

        # ---- [DEBUG] 每个位置的KL散度统计 ----
        print(f"    kl_div per position range: [{kl_div.min().item():.6f}, {kl_div.max().item():.6f}]")
        print(f"    kl_div per position mean: {kl_div.mean().item():.6f}")
        print(f"    kl_div > 10 positions: {(kl_div > 10).sum().item()}/{kl_div.numel()}")
        print(f"    kl_div > 100 positions: {(kl_div > 100).sum().item()}/{kl_div.numel()}")

        masked_kl = kl_div * loss_mask
        valid_kl_values = masked_kl[loss_mask.bool()]
        if valid_kl_values.numel() > 0:
            print(f"    valid kl_div values mean: {valid_kl_values.mean().item():.6f}")
            print(f"    num_valid_tokens: {loss_mask.sum().item():.0f}")

        loss = masked_kl.sum() / (loss_mask.sum() + 1e-8) * (self.temperature ** 2)
        print(f"    >>> Common-KL Loss: {loss.item():.6f}")

        return loss

    def _compute_uld_loss(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        unmatched_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        student_vocab: int,
        teacher_vocab: int,
    ) -> Optional[torch.Tensor]:
        """Compute ULD (Unmatched Logit Distillation) loss.

        For student tokens that have NO mapping in the projection matrix (W[s,:]=0),
        compute a standard KL divergence to provide a regularization signal.

        Since teacher and student vocabularies differ, we compute ULD as follows:
        1. Gather student probabilities only for unmatched tokens
        2. For positions where teacher has non-zero probability, compute KL divergence
           between teacher distribution and a reference distribution derived from
           the unmatched student probabilities

        If there are no unmatched tokens or no valid positions, return None.

        Args:
            student_probs: [batch, seq_len, student_vocab]
            teacher_probs: [batch, seq_len, teacher_vocab]
            unmatched_mask: [student_vocab] - True for tokens with no mapping
            loss_mask: [batch, seq_len]
            batch_size, seq_len, student_vocab, teacher_vocab: dimensions

        Returns:
            Scalar ULD loss, or None if no unmatched tokens exist.
        """
        unmatched_ids = torch.where(unmatched_mask)[0]
        if len(unmatched_ids) == 0:
            return None

        # Gather student probabilities for unmatched tokens only
        # unmatched_student_probs: [batch, seq_len, num_unmatched]
        unmatched_student_probs = student_probs.index_select(dim=-1, index=unmatched_ids)

        # For ULD, we compute KL(teacher || student_reference) in a reduced space.
        # Since teacher and student have different vocabularies, we use the teacher's
        # non-zero probability positions as the reference distribution and compare
        # with the normalized student distribution over unmatched tokens.
        #
        # Approach: Treat the unmatched student probabilities as a sub-distribution.
        # Normalize them to sum to 1 over the unmatched tokens.
        unmatched_sum = unmatched_student_probs.sum(dim=-1, keepdim=True) + 1e-8
        normalized_unmatched = unmatched_student_probs / unmatched_sum

        # For the teacher side, we use the full teacher distribution but only
        # compute KL at positions where the teacher has non-zero probability.
        teacher_nonzero_mask = teacher_probs > 0

        # Create a reference distribution from teacher: normalize over non-zero positions
        teacher_reference = teacher_probs * teacher_nonzero_mask.float()
        teacher_reference = teacher_reference / teacher_reference.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # For ULD, we compute a simplified KL divergence:
        # We project the unmatched student distribution onto the teacher vocabulary
        # by using a uniform mapping (each unmatched token contributes equally)
        num_unmatched = len(unmatched_ids)
        # Average probability of unmatched tokens per position
        avg_unmatched_prob = unmatched_student_probs.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]

        # Create a student reference distribution in teacher vocab space
        # by distributing the average unmatched probability uniformly across
        # teacher non-zero positions
        student_reference = avg_unmatched_prob.expand_as(teacher_probs) * teacher_nonzero_mask.float()
        student_reference = student_reference / student_reference.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute KL(teacher || student_reference) only at non-zero teacher positions
        log_student_ref = torch.log(student_reference + 1e-8)

        # ---- [DEBUG] ULD KL散度计算前统计 ----
        print(f"\n  [ULD-KL DEBUG]")
        print(f"    num_unmatched_tokens: {num_unmatched}/{student_vocab}")
        print(f"    unmatched_student_probs range: [{unmatched_student_probs.min().item():.6f}, {unmatched_student_probs.max().item():.6f}]")
        print(f"    unmatched_sum per position: mean={unmatched_sum.mean().item():.6f}")
        print(f"    avg_unmatched_prob range: [{avg_unmatched_prob.min().item():.6f}, {avg_unmatched_prob.max().item():.6f}]")
        print(f"    teacher_reference range: [{teacher_reference.min().item():.6f}, {teacher_reference.max().item():.6f}]")
        print(f"    teacher_reference sum per position: mean={teacher_reference.sum(dim=-1).mean().item():.6f}")
        print(f"    student_reference range: [{student_reference.min().item():.6f}, {student_reference.max().item():.6f}]")
        print(f"    student_reference sum per position: mean={student_reference.sum(dim=-1).mean().item():.6f}")
        print(f"    student_reference == 0 ratio: {(student_reference == 0).float().mean().item():.4f}")
        print(f"    log(student_reference + 1e-8) range: [{log_student_ref.min().item():.4f}, {log_student_ref.max().item():.4f}]")

        kl_contrib = teacher_reference * (
            torch.log(teacher_reference + 1e-8) - log_student_ref
        )

        # ---- [DEBUG] KL贡献值统计 ----
        print(f"    kl_contrib (before mask) range: [{kl_contrib.min().item():.6f}, {kl_contrib.max().item():.6f}]")
        print(f"    kl_contrib (before mask) mean: {kl_contrib.mean().item():.6f}")
        print(f"    kl_contrib > 10 ratio: {(kl_contrib > 10).float().mean().item():.6f}")
        print(f"    kl_contrib is NaN: {torch.isnan(kl_contrib).any().item()}")
        print(f"    kl_contrib is Inf: {torch.isinf(kl_contrib).any().item()}")

        kl_contrib = kl_contrib * teacher_nonzero_mask
        kl_div = kl_contrib.sum(dim=-1)  # [batch, seq_len]

        # ---- [DEBUG] 每个位置的KL散度统计 ----
        print(f"    kl_div per position range: [{kl_div.min().item():.6f}, {kl_div.max().item():.6f}]")
        print(f"    kl_div per position mean: {kl_div.mean().item():.6f}")

        # Apply loss mask
        masked_kl = kl_div * loss_mask
        num_valid = loss_mask.sum()

        if num_valid.item() == 0:
            print(f"    >>> ULD Loss: None (no valid tokens)")
            return None

        loss = masked_kl.sum() / (num_valid + 1e-8)
        print(f"    num_valid_tokens: {num_valid.item():.0f}")
        print(f"    >>> ULD Loss: {loss.item():.6f}")
        return loss

    # =========================================================================
    # Top-k vocabulary selection (NeMo-RL style)
    # =========================================================================

    @staticmethod
    def _select_teacher_topk_indices(
        teacher_values: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Select top-k teacher vocabulary indices per batch following NeMo-RL.

        Importance metric: flatten (B * T), then per-vocab max(value).
        This selects the k most "active" tokens across the entire batch.
        Works with both raw logits and probabilities (monotonic transform).

        Args:
            teacher_values: [batch, seq_len, vocab_size] teacher logits or probs.
            k: Number of top-k indices to select.

        Returns:
            top_indices: [k] sorted teacher vocabulary indices.
        """
        vocab_size = teacher_values.shape[-1]
        k = min(k, vocab_size)
        with torch.no_grad():
            teacher_flat = teacher_values.reshape(-1, vocab_size)
            importance = teacher_flat.max(dim=0).values  # [vocab_size]
            top_indices = torch.topk(importance, k=k, dim=-1).indices
            return top_indices.sort().values  # [k]

    # =========================================================================
    # Sequence alignment utilities
    # =========================================================================

    @staticmethod
    def _decode_tokens(tokenizer, token_ids):
        """Decode tokens to character-level spans.

        For each token position, returns the character-level text span
        by computing the difference between cumulative decodings.

        Args:
            tokenizer: HuggingFace tokenizer.
            token_ids: List of token IDs.

        Returns:
            List of strings, each being the text span for the corresponding token.
        """
        pieces = []
        prev = ""
        for k in range(len(token_ids)):
            cur = tokenizer.decode(token_ids[:k + 1], skip_special_tokens=False)
            pieces.append(cur[len(prev):])
            prev = cur
        return pieces

    def _align_sequences_by_text(
        self,
        student_input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        teacher_index: int = 0,
    ) -> torch.Tensor:
        """Align student and teacher token sequences by character-level text matching.

        Uses greedy substring matching to determine which student position(s)
        correspond to each teacher position.

        Args:
            student_input_ids: [batch, N] student token IDs.
            teacher_input_ids: [batch, M] teacher token IDs.
            teacher_index: Index of the teacher model (for tokenizer selection).

        Returns:
            teacher_to_student_map: [batch, M] mapping from teacher position to
                student position. -1 means no matching student position found.
        """
        batch_size = student_input_ids.shape[0]
        teacher_seq_len = teacher_input_ids.shape[1]
        teacher_tokenizer = self.teacher_tokenizer_group[teacher_index]

        # Decode all sequences to character spans
        teacher_maps = []
        for b in range(batch_size):
            student_ids = student_input_ids[b].tolist()
            teacher_ids = teacher_input_ids[b].tolist()

            # Filter out padding/ignore tokens
            student_ids = [sid for sid in student_ids if sid != -100 and sid != 0]
            teacher_ids = [tid for tid in teacher_ids if tid != -100 and tid != 0]

            if not student_ids or not teacher_ids:
                teacher_maps.append(torch.full((teacher_seq_len,), -1, dtype=torch.long))
                continue

            # Decode to character spans
            student_pieces = self._decode_tokens(self.student_tokenizer, student_ids)
            teacher_pieces = self._decode_tokens(teacher_tokenizer, teacher_ids)

            # Greedy matching: for each teacher position, find corresponding student positions
            teacher_to_student = torch.full((len(teacher_ids),), -1, dtype=torch.long)
            s_idx = 0
            t_idx = 0

            while t_idx < len(teacher_pieces) and s_idx < len(student_pieces):
                # Accumulate text from current positions
                teacher_text = teacher_pieces[t_idx]
                student_text = student_pieces[s_idx]

                if teacher_text == student_text:
                    # Exact match at single position
                    teacher_to_student[t_idx] = s_idx
                    t_idx += 1
                    s_idx += 1
                elif len(teacher_text) < len(student_text):
                    # Teacher text is shorter, teacher may need multiple positions
                    # Map this teacher position to current student position
                    teacher_to_student[t_idx] = s_idx
                    t_idx += 1
                    # Don't advance student, it may cover multiple teacher positions
                else:
                    # Student text is shorter, student may need multiple positions
                    # Map this teacher position to current student position
                    teacher_to_student[t_idx] = s_idx
                    s_idx += 1
                    # Don't advance teacher, it may cover multiple student positions

            # Pad to full sequence length
            full_map = torch.full((teacher_seq_len,), -1, dtype=torch.long)
            full_map[:len(teacher_to_student)] = teacher_to_student
            teacher_maps.append(full_map)

        return torch.stack(teacher_maps, dim=0)

    # =========================================================================
    # Public utility methods
    # =========================================================================

    # =========================================================================
    # Serialization support (for Ray distributed training)
    # =========================================================================

    def __getstate__(self):
        """Custom serialization for Ray compatibility.

        Excludes non-picklable objects:
            - _build_lock: threading.Lock is not serializable
            - _auto_select_loss_type: bound method, not needed after construction

        Note: tokenizers are serializable and are kept for rebuilding projection matrices.
        """
        state = self.__dict__.copy()
        # Remove non-serializable objects
        state.pop('_build_lock', None)
        # Remove bound methods that are not needed after construction
        state.pop('_auto_select_loss_type', None)
        return state

    def __setstate__(self, state):
        """Custom deserialization for Ray compatibility.

        Recreates non-serializable objects that were excluded in __getstate__.
        """
        self.__dict__.update(state)
        # Recreate the lock
        self._build_lock = threading.Lock()

    def get_projection_matrix(self, teacher_index: int = 0) -> Optional[torch.Tensor]:
        """Return the dense projection matrix W for a specific teacher.

        Note: Returns None for large vocabularies (sparse format is used).
        Use get_sparse_projection_data() for sparse representation.
        """
        self._ensure_projection_matrices_built()
        if teacher_index >= len(self.projection_matrices):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")
        return self.projection_matrices[teacher_index]

    def get_sparse_projection_data(self, teacher_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sparse projection matrix data in COO format.

        Returns:
            Tuple of (student_indices, teacher_indices, values).
        """
        self._ensure_projection_matrices_built()
        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")
        return (
            self.projection_student_indices_list[teacher_index],
            self.projection_teacher_indices_list[teacher_index],
            self.projection_values_list[teacher_index],
        )

    def get_mapping_statistics(self, teacher_index: int = 0) -> Dict:
        """Return statistics about the projection matrix for a specific teacher.

        Returns:
            Dict with mapping statistics including matched/unmatched counts.
        """
        self._ensure_projection_matrices_built()
        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")

        student_indices = self.projection_student_indices_list[teacher_index]
        values = self.projection_values_list[teacher_index].float()

        nnz = student_indices.numel()
        total_elements = self.student_vocab_size * self.teacher_vocab_sizes[teacher_index]
        sparsity = 1.0 - (nnz / total_elements)

        # Count exact matches (weight == 1)
        exact_match_mask = (values == 1.0)
        exact_matched_students = student_indices[exact_match_mask].unique()
        exact_matched = exact_matched_students.numel()

        # Count multi-token matches (0 < weight < 1)
        multi_token_mask = (values > 0) & (values < 1.0)
        multi_token_students = student_indices[multi_token_mask].unique()
        # Exclude students that already have exact matches
        multi_token_students = multi_token_students[
            ~multi_token_students.unsqueeze(1).eq(exact_matched_students.unsqueeze(0)).any(dim=1)
        ]
        multi_token_matched = multi_token_students.numel()

        # Count unmatched: total - exact_matched - multi_token_matched
        unmatched = self.student_vocab_size - exact_matched - multi_token_matched

        return {
            'total_student_tokens': self.student_vocab_size,
            'exact_matched': exact_matched,
            'multi_token_matched': multi_token_matched,
            'unmatched': unmatched,
            'sparsity': sparsity,
        }
