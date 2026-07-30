# Copyright (c) ModelScope Contributors. All rights reserved.
"""
CrossTokenLoss: Projection-Guided Cross-Tokenizer Knowledge Distillation Loss.

This module implements the X-Token approach with two loss modes:

1. **P-KL (Projection-KL)**: Projects student distribution to teacher vocabulary
   space via a sparse projection matrix W, then computes chunk-averaged KL
   divergence over a microbatch-global top-k teacher subset. This is the
   NeMo-style P-KL path.

2. **H-KL (Heuristic-KL)**: Partitions the student vocabulary into *common*
   (exact-token-mapped) and *uncommon* (no direct mapping) subsets. Computes
   KL divergence on common tokens, L1 loss on uncommon tokens, and ULD
   (Unmatched Logit Distillation) for student tokens with no projection
   mapping.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)

Key differences from CTKDLoss:
    1. P-KL mode adds chunk-averaged KL over top-k teacher subset (NeMo-style).
    2. H-KL mode adds common/uncommon partition + ULD loss component.
    3. Supports both full teacher logits (TransformersModel) and top-k
       logprobs (vLLM sampler).
    4. Multi-teacher aggregation with static weights.
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

# Global cache for projection matrices and exact token maps
_PROJECTION_MATRIX_CACHE = {}
_EXACT_TOKEN_MAP_CACHE = {}


class CrossTokenLoss(Loss):
    """Cross-Tokenizer Knowledge Distillation Loss using X-Token projection.

    Supports two loss modes:
    - ``'pkl'``: NeMo-style Projection-KL with top-k subset.
    - ``'hkl'``: Gold-loss with common/uncommon partition + ULD.

    Teacher data can be provided as:
    - ``teacher_logits_group``: list of [B, T_t, V_t] full logits
    - ``teacher_topk_logprobs_group`` + ``teacher_topk_indices_group``: top-k format
    - ``teacher_input_ids_group``: optional, for sequence alignment

    Args:
        student_tokenizer: Student model tokenizer.
        teacher_tokenizer_group: List of teacher tokenizers.
        teacher_weights: Per-teacher static weights (default: equal).
        max_length: Maximum span length L for multi-token matching.
        beta: Base weight β for projection.
        gamma: Decay rate γ for multi-token weights.
        loss_type: 'pkl' for projection-KL or 'hkl' for heuristic-KL.
        temperature: Softmax temperature for KL divergence.
        gamma_kl: Weight for common-KL term in H-KL mode.
        gamma_uld: Weight for ULD term in H-KL mode.
        vocab_topk: Top-k vocab size for teacher logits subset in P-KL.
        uncommon_topk: Top-k for uncommon L1 loss in H-KL mode.
        reverse_kl: If True, compute KL(student || teacher).
        exact_token_match_only: Only use exact token matches (no multi-token).
        kl_loss_weight: Scalar multiplier on distillation loss.
        ce_loss_weight: Scalar multiplier on CE auxiliary loss.
        dynamic_loss_scaling: Rescale KD term to match CE magnitude each step.
        device: Device for projection matrices.
    """

    require_logits = True

    def __init__(
        self,
        student_tokenizer: 'PreTrainedTokenizer',
        teacher_tokenizer_group: list,
        teacher_weights: Optional[list] = None,
        max_length: int = 4,
        beta: float = 0.95,
        gamma: float = 0.1,
        loss_type: str = 'pkl',
        temperature: float = 1.0,
        gamma_kl: float = 1.0,
        gamma_uld: float = 0.5,
        vocab_topk: int = 512,
        uncommon_topk: int = 8192,
        reverse_kl: bool = False,
        exact_token_match_only: bool = False,
        kl_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
        dynamic_loss_scaling: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer_group = teacher_tokenizer_group
        self.num_teachers = len(teacher_tokenizer_group)

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
        self.loss_type = loss_type
        self.temperature = temperature
        self.gamma_kl = gamma_kl
        self.gamma_uld = gamma_uld
        self.vocab_topk = vocab_topk
        self.uncommon_topk = uncommon_topk
        self.reverse_kl = reverse_kl
        self.exact_token_match_only = exact_token_match_only
        self.kl_loss_weight = kl_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.dynamic_loss_scaling = dynamic_loss_scaling

        if loss_type not in ('pkl', 'hkl'):
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'pkl' or 'hkl'.")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = torch.device('npu:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.student_vocab_size = len(student_tokenizer)
        self.teacher_vocab_sizes = [len(tok) for tok in teacher_tokenizer_group]

        # Lazy initialization
        self._projection_matrices_built = False
        self.projection_student_indices_list: list = []
        self.projection_teacher_indices_list: list = []
        self.projection_values_list: list = []

        # H-KL: exact token maps (common/uncommon partition)
        self._exact_maps_built = False
        self._common_student_indices_list: list = []
        self._common_teacher_indices_list: list = []
        self._uncommon_student_indices_list: list = []
        self._uncommon_teacher_indices_list: list = []

    # ------------------------------------------------------------------
    # Projection matrix building (same as CTKDLoss)
    # ------------------------------------------------------------------

    def _ensure_projection_matrices_built(self):
        """Lazy-build projection matrices with caching."""
        if self._projection_matrices_built:
            return
        if not hasattr(self, '_build_lock'):
            self._build_lock = threading.Lock()
        with self._build_lock:
            if self._projection_matrices_built:
                return
            import time
            start_time = time.perf_counter()

            cache_key = self._generate_cache_key()
            if cache_key in _PROJECTION_MATRIX_CACHE:
                cached = _PROJECTION_MATRIX_CACHE[cache_key]
                self.projection_student_indices_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['student_indices']
                ]
                self.projection_teacher_indices_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['teacher_indices']
                ]
                self.projection_values_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['values']
                ]
            else:
                for i, teacher_tok in enumerate(self.teacher_tokenizer_group):
                    self._build_projection_matrix_for_teacher(teacher_tok, i)
                _PROJECTION_MATRIX_CACHE[cache_key] = {
                    'student_indices': self.projection_student_indices_list,
                    'teacher_indices': self.projection_teacher_indices_list,
                    'values': self.projection_values_list,
                }

            # Build exact token maps for H-KL
            if self.loss_type == 'hkl':
                self._build_exact_token_maps()

            elapsed = time.perf_counter() - start_time
            print(f"[CrossToken] Projection matrices built in {elapsed:.2f}s")
            self._projection_matrices_built = True

    def _build_exact_token_maps(self):
        """Build common/uncommon token partitions for H-KL mode.

        Common: tokens with exact text match between student and teacher.
        Uncommon: all other tokens.
        """
        self._common_student_indices_list = []
        self._common_teacher_indices_list = []
        self._uncommon_student_indices_list = []
        self._uncommon_teacher_indices_list = []

        # Build student text -> id mapping
        student_vocab = self.student_tokenizer.get_vocab()
        student_id_to_text = {}
        for token_str, token_id in student_vocab.items():
            student_id_to_text[token_id] = self.student_tokenizer.decode(
                [token_id], skip_special_tokens=False
            ).strip()

        for i, teacher_tok in enumerate(self.teacher_tokenizer_group):
            teacher_vocab = teacher_tok.get_vocab()
            teacher_id_to_text = {}
            for token_str, token_id in teacher_vocab.items():
                teacher_id_to_text[token_id] = teacher_tok.decode(
                    [token_id], skip_special_tokens=False
                ).strip()

            # Build text -> teacher_id mapping
            teacher_text_to_id = {}
            for token_id in range(len(teacher_tok)):
                token_text = teacher_tok.decode([token_id], skip_special_tokens=False).strip()
                teacher_text_to_id[token_text] = token_id

            # Find common tokens
            common_s = []
            common_t = []
            uncommon_s = []
            for s_id in range(self.student_vocab_size):
                if s_id in student_id_to_text:
                    s_text = student_id_to_text[s_id]
                    if s_text in teacher_text_to_id:
                        common_s.append(s_id)
                        common_t.append(teacher_text_to_id[s_text])
                    else:
                        uncommon_s.append(s_id)

            # Uncommon teacher tokens (not mapped to any student)
            mapped_teacher_ids = set(common_t)
            uncommon_t = [t_id for t_id in range(len(teacher_tok))
                          if t_id not in mapped_teacher_ids]

            self._common_student_indices_list.append(
                torch.tensor(common_s, dtype=torch.long, device=self.device)
            )
            self._common_teacher_indices_list.append(
                torch.tensor(common_t, dtype=torch.long, device=self.device)
            )
            self._uncommon_student_indices_list.append(
                torch.tensor(uncommon_s, dtype=torch.long, device=self.device)
            )
            self._uncommon_teacher_indices_list.append(
                torch.tensor(uncommon_t, dtype=torch.long, device=self.device)
            )

            n_common = len(common_s)
            n_uncommon_s = len(uncommon_s)
            n_uncommon_t = len(uncommon_t)
            print(f"[CrossToken] Teacher {i}: {n_common} common tokens, "
                  f"{n_uncommon_s} uncommon student, {n_uncommon_t} uncommon teacher")

    def _generate_cache_key(self) -> str:
        """Generate cache key from tokenizer configurations."""
        config_data = {
            'student_vocab': self.student_tokenizer.get_vocab(),
            'teacher_vocabs': [t.get_vocab() for t in self.teacher_tokenizer_group],
            'max_length': self.max_length,
            'beta': self.beta,
            'gamma': self.gamma,
        }
        config_bytes = pickle.dumps(config_data)
        return hashlib.md5(config_bytes).hexdigest()

    def _build_projection_matrix_for_teacher(self, teacher_tokenizer, teacher_index):
        """Build sparse projection matrix W in COO format."""
        student_indices = []
        teacher_indices = []
        values = []
        matched_student_ids = set()

        teacher_token_text_to_id = {}
        for token_id in range(len(teacher_tokenizer)):
            token_text = teacher_tokenizer.decode([token_id], skip_special_tokens=False).strip()
            teacher_token_text_to_id[token_text] = token_id

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

        for student_id in range(self.student_vocab_size):
            if student_id in matched_student_ids:
                continue
            text = self.student_tokenizer.decode([student_id], skip_special_tokens=False)
            if not text or not text.strip():
                continue
            teacher_token_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
            seq_length = len(teacher_token_ids)
            if 0 < seq_length < self.max_length:
                for i, t_id in enumerate(teacher_token_ids):
                    weight = self.beta * (self.gamma ** i)
                    student_indices.append(student_id)
                    teacher_indices.append(t_id)
                    values.append(weight)

        student_tensor = torch.tensor(student_indices, dtype=torch.long, device=self.device)
        teacher_tensor = torch.tensor(teacher_indices, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.projection_student_indices_list.append(student_tensor)
        self.projection_teacher_indices_list.append(teacher_tensor)
        self.projection_values_list.append(values_tensor)

        print(f"[CrossToken] Teacher {teacher_index}: {len(student_indices)} mappings")

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        """Compute CrossToken distillation loss.

        Args:
            inputs: Dict with 'labels' [B, S] (with -100 for padding).
            outputs: Dict with 'logits' [B, S, V_s] from student model.
            **kwargs: Teacher data.

        Teacher data kwargs:
            teacher_logits_group: list of [B, T, V_t] full logits
            teacher_topk_logprobs_group: list of [B, T, topk] logprobs
            teacher_topk_indices_group: list of [B, T, topk] indices
            teacher_labels: list of [B, S] labels for loss mask

        Returns:
            LossOutput with the computed loss and number of tokens.
        """
        self._ensure_projection_matrices_built()

        # Get teacher data
        teacher_logits_group = kwargs.get('teacher_logits_group')
        if teacher_logits_group is None:
            teacher_logits_group = outputs.get('teacher_logits_group')

        teacher_topk_logprobs_group = kwargs.get('teacher_topk_logprobs_group')
        teacher_topk_indices_group = kwargs.get('teacher_topk_indices_group')

        # Convert top-k format to full logits-like if needed
        if teacher_logits_group is None and teacher_topk_logprobs_group is not None \
                and teacher_topk_indices_group is not None:
            teacher_logits_group = self._topk_to_full(
                teacher_topk_logprobs_group, teacher_topk_indices_group
            )

        if teacher_logits_group is None:
            raise ValueError(
                "No teacher logits provided. Pass teacher_logits_group or "
                "(teacher_topk_logprobs_group + teacher_topk_indices_group)."
            )
        if len(teacher_logits_group) != self.num_teachers:
            raise ValueError(
                f"Got {len(teacher_logits_group)} teacher logits, expected {self.num_teachers}."
            )

        # Get labels for loss mask
        student_labels = inputs.get('labels')
        if student_labels is None:
            raise ValueError("labels not found in inputs")

        teacher_labels_group = kwargs.get('teacher_labels')
        if teacher_labels_group is not None and len(teacher_labels_group) > 0:
            labels = teacher_labels_group[0].to(self.device)
        else:
            labels = student_labels

        student_logits = outputs.get('logits')
        if student_logits is None:
            raise ValueError("logits not found in outputs")

        # Compute loss per teacher
        total_kd = torch.tensor(0.0, device=student_logits.device)
        ce_loss = self._compute_ce(student_logits, labels)
        teacher_metrics = []

        for i in range(self.num_teachers):
            t_logits = teacher_logits_group[i]
            weight = self.teacher_weights[i]

            if self.loss_type == 'pkl':
                kd, metrics = self._compute_pkl(
                    student_logits, t_logits, labels, i
                )
            elif self.loss_type == 'hkl':
                kd, metrics = self._compute_hkl(
                    student_logits, t_logits, labels, i
                )
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            total_kd = total_kd + weight * kd
            teacher_metrics.append({'index': i, 'weight': weight, **metrics})

        # Combine KD + CE
        if self.dynamic_loss_scaling:
            kd_detached = total_kd.detach().abs()
            ce_detached = ce_loss.detach().abs()
            kl_scale = torch.where(
                kd_detached > 0, ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            loss = kl_scale * total_kd + ce_loss
        else:
            loss = self.kl_loss_weight * total_kd + self.ce_loss_weight * ce_loss

        # Compute number of valid tokens
        loss_mask = (labels != -100).float()
        num_tokens = int(loss_mask.sum().item())

        # Print metrics
        print(f"\n=== CrossTokenLoss (type={self.loss_type}) ===")
        print(f"  KD loss: {total_kd.item():.6f}  CE loss: {ce_loss.item():.6f}")
        print(f"  Total loss: {loss.item():.6f}  num_tokens: {num_tokens}")
        for m in teacher_metrics:
            idx = m['index']
            print(f"  Teacher {idx} (w={m['weight']:.3f}): kd={m['kd_loss']:.6f}")
            if 'proj_accuracy' in m:
                print(f"    proj_acc={m['proj_accuracy']:.4f}")
            if 'kl_common' in m:
                print(f"    kl_common={m['kl_common']:.6f}  "
                      f"l1_uncommon={m.get('l1_uncommon', 0):.6f}  "
                      f"uld={m.get('uld_loss', 0):.6f}")
        print("=" * 50)
        return LossOutput(loss=loss, num_tokens=num_tokens)

    def _topk_to_full(self, topk_logprobs_group, topk_indices_group):
        """Convert vLLM top-k logprobs to full-vocab probability tensors."""
        full_group = []
        for i, (topk_lp, topk_idx) in enumerate(
            zip(topk_logprobs_group, topk_indices_group)
        ):
            vocab_size = self.teacher_vocab_sizes[i]
            batch_size, seq_len, topk = topk_lp.shape
            probs_full = torch.zeros(
                batch_size, seq_len, vocab_size,
                dtype=topk_lp.dtype, device=self.device,
            )
            topk_probs = torch.exp(topk_lp.to(self.device))
            topk_idx_dev = topk_idx.to(self.device)
            probs_full.scatter_(dim=2, index=topk_idx_dev, src=topk_probs)
            full_group.append(probs_full)
        return full_group

    def _compute_ce(self, student_logits, labels):
        """Compute auxiliary next-token cross-entropy loss."""
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)
        loss_mask = (shift_labels != -100).float()
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        ce_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction='none',
        ).reshape(shift_labels.shape)
        return (ce_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # P-KL mode
    # ------------------------------------------------------------------

    def _compute_pkl(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        labels: torch.Tensor,
        teacher_index: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """NeMo-style projection-KL loss.

        Steps:
        1. Shift logits for next-token prediction.
        2. Compute student probs with temperature, project to teacher vocab.
        3. Select microbatch-global top-k from teacher.
        4. Slice both sides to top-k and renormalize.
        5. Compute per-position KL, mask, and mean.
        6. Scale by T^2.
        """
        T = self.temperature
        eps = 1e-10

        # Shift for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_probs[..., :-1, :].contiguous().to(student_logits.device)

        # Align sequence lengths
        stu_seq_len = shift_student_logits.shape[1]
        tea_seq_len = shift_teacher.shape[1]
        min_seq_len = min(stu_seq_len, tea_seq_len)
        shift_student_logits = shift_student_logits[:, :min_seq_len, :]
        shift_teacher = shift_teacher[:, :min_seq_len, :]

        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)
        shift_labels = shift_labels[:, :min_seq_len]
        loss_mask = (shift_labels != -100).float()

        batch_size, seq_len, _ = shift_student_logits.shape
        # Tokenizer vocab size (used to size the projection matrix).
        # The model's lm_head may have a larger out_features (HF padding).
        tkr_vocab_size = self.teacher_vocab_sizes[teacher_index]
        model_vocab_size = shift_teacher.shape[-1]

        # Student log-probs with temperature
        student_log_probs = F.log_softmax(shift_student_logits / T, dim=-1)
        student_probs = student_log_probs.exp()

        # Project student probs to teacher vocab via sparse W
        projected = self._project_student_probs(
            student_probs, teacher_index, tkr_vocab_size
        )

        # Slice teacher tensor to the tokenizer vocab width — the projection
        # matrix is sized to len(tokenizer), while the model may pad
        # lm_head.out_features beyond that (e.g. Qwen3: tokenizer 151669,
        # lm_head 151936). The padded columns are not real tokens and have
        # no projection entries, so they are discarded.
        if model_vocab_size > tkr_vocab_size:
            shift_teacher = shift_teacher[..., :tkr_vocab_size]

        # Microbatch-global top-k from teacher logits
        # If teacher_probs are actually probabilities (not logits), convert back
        if shift_teacher.max() <= 1.0 and shift_teacher.sum(dim=-1).min() > 0.5:
            teacher_logits_approx = torch.log(shift_teacher.clamp(min=eps))
        else:
            teacher_logits_approx = shift_teacher

        k = min(self.vocab_topk, tkr_vocab_size, projected.shape[-1])
        teacher_flat = teacher_logits_approx.reshape(-1, tkr_vocab_size)
        importance = teacher_flat.max(dim=0).values  # [V_t]
        _, topk_idx = torch.topk(importance, k=k)
        topk_idx = topk_idx.sort().values  # [k]

        # Slice to top-k subset.
        # projected_k comes from projected student probs (non-negative).
        projected_k = projected[..., topk_idx]  # [B, S, k]
        # teacher_logits_approx contains raw logits; use log_softmax to get
        # proper log-probs, exactly as NeMo does in _compute_p_kl.
        teacher_log_probs_k = torch.log_softmax(
            teacher_logits_approx[..., topk_idx] / T, dim=-1
        )  # [B, S, k]

        # Renormalize projected student probs within the top-k subset.
        projected_k = projected_k / (projected_k.sum(dim=-1, keepdim=True) + eps)
        log_projected_k = (projected_k + eps).log()

        # Teacher is already log-probs from log_softmax.
        log_teacher_k = teacher_log_probs_k
        teacher_k = log_teacher_k.exp()  # for accuracy computation below

        if self.reverse_kl:
            per_pos_kl = F.kl_div(
                log_teacher_k, log_projected_k,
                reduction='none', log_target=True,
            ).sum(dim=-1)
        else:
            per_pos_kl = F.kl_div(
                log_projected_k, log_teacher_k,
                reduction='none', log_target=True,
            ).sum(dim=-1)

        masked_kl = (per_pos_kl * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        kd_loss = masked_kl * T * T

        # Projection accuracy: argmax match on top-k subset
        with torch.no_grad():
            proj_top1 = projected_k.argmax(dim=-1)  # [B, S]
            teach_top1 = teacher_k.argmax(dim=-1)  # [B, S]
            matches = (proj_top1 == teach_top1) & loss_mask.bool()
            proj_acc = matches.sum().float() / loss_mask.sum().clamp(min=1.0)

        metrics = {
            'kd_loss': kd_loss.item(),
            'proj_accuracy': proj_acc.item(),
        }
        return kd_loss, metrics

    def _project_student_probs(
        self,
        student_probs: torch.Tensor,
        teacher_index: int,
        teacher_vocab_size: int,
    ) -> torch.Tensor:
        """Project student probs [B, S, V_s] to teacher vocab [B, S, V_t] via sparse W."""
        batch_size, seq_len, _ = student_probs.shape

        s_idx = self.projection_student_indices_list[teacher_index].to(student_probs.device)
        t_idx = self.projection_teacher_indices_list[teacher_index].to(student_probs.device)
        values = self.projection_values_list[teacher_index].to(student_probs.device).float()

        # Filter indices to valid range
        valid_mask = t_idx < teacher_vocab_size
        s_idx = s_idx[valid_mask]
        t_idx = t_idx[valid_mask]
        values = values[valid_mask]

        if s_idx.numel() == 0:
            return torch.zeros(batch_size, seq_len, teacher_vocab_size,
                               device=student_probs.device, dtype=student_probs.dtype)

        # Gather student probs at non-zero projection entries
        selected = student_probs.index_select(dim=-1, index=s_idx)  # [B, S, N]
        weighted = selected * values.unsqueeze(0).unsqueeze(0)  # [B, S, N]

        # Scatter-add to teacher vocab
        projected = torch.zeros(
            batch_size, seq_len, teacher_vocab_size,
            device=student_probs.device, dtype=student_probs.dtype,
        )
        expanded_idx = t_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        projected.scatter_add_(dim=2, index=expanded_idx, src=weighted)

        return projected

    # ------------------------------------------------------------------
    # H-KL mode (gold loss with common/uncommon + ULD)
    # ------------------------------------------------------------------

    def _compute_hkl(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        labels: torch.Tensor,
        teacher_index: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """H-KL loss: common-KL + uncommon-L1 + ULD.

        Steps:
        1. Shift for next-token, align sequences.
        2. Compute chunk-averaged logprobs for student and teacher.
        3. Split into common (KL) and uncommon (L1) partitions.
        4. ULD: KL in student vocab space for unmatched student tokens.
        5. Weighted sum: gamma_kl * common_KL + (1-gamma_kl-gamma_uld) * L1 + gamma_uld * ULD.
        """
        T = self.temperature
        eps = 1e-10

        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_probs[..., :-1, :].contiguous().to(student_logits.device)

        stu_seq_len = shift_student_logits.shape[1]
        tea_seq_len = shift_teacher.shape[1]
        min_seq_len = min(stu_seq_len, tea_seq_len)
        shift_student_logits = shift_student_logits[:, :min_seq_len, :]
        shift_teacher = shift_teacher[:, :min_seq_len, :]

        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)
        shift_labels = shift_labels[:, :min_seq_len]
        loss_mask = (shift_labels != -100).float()

        # Tokenizer vocab size (sizes the exact-token map).  The model's
        # lm_head may have a larger out_features (HF padding), so slice
        # the teacher tensor to the tokenizer width to align with the
        # common/uncommon index tensors.
        tkr_vocab_size = self.teacher_vocab_sizes[teacher_index]
        model_vocab_size = shift_teacher.shape[-1]
        if model_vocab_size > tkr_vocab_size:
            shift_teacher = shift_teacher[..., :tkr_vocab_size]

        # Student log-probs on full vocab
        student_log_probs = F.log_softmax(shift_student_logits / T, dim=-1)
        student_probs = student_log_probs.exp()

        # Teacher log-probs (teacher_probs may be probs or raw logits)
        if shift_teacher.max() > 1.0 or shift_teacher.sum(dim=-1).min() < 0.5:
            teacher_log_probs = F.log_softmax(shift_teacher / T, dim=-1)
        else:
            teacher_log_probs = (shift_teacher + eps).log()

        # Get exact token map
        common_s = self._common_student_indices_list[teacher_index]
        common_t = self._common_teacher_indices_list[teacher_index]
        uncommon_s = self._uncommon_student_indices_list[teacher_index]
        uncommon_t = self._uncommon_teacher_indices_list[teacher_index]

        # --- Common KL ---
        common_kl = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        if common_s.numel() > 0:
            s_common = student_log_probs[..., common_s.to(student_logits.device)]  # [B, S, N_c]
            t_common = teacher_log_probs[..., common_t.to(student_logits.device)]  # [B, S, N_c]

            if self.reverse_kl:
                kl_elem = F.kl_div(t_common, s_common, reduction='none', log_target=True)
            else:
                kl_elem = F.kl_div(s_common, t_common, reduction='none', log_target=True)

            kl_per_pos = kl_elem.sum(dim=-1)  # [B, S]
            common_kl = (kl_per_pos * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        # --- Uncommon L1 ---
        l1_uncommon = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        uncommon_topk = self.uncommon_topk

        if (uncommon_s.numel() > 0 or uncommon_t.numel() > 0) and loss_mask.sum() > 0:
            s_unc = student_probs[..., uncommon_s.to(student_logits.device)]  # [B, S, N_us]
            t_unc = teacher_log_probs.exp()[..., uncommon_t.to(student_logits.device)]  # [B, S, N_ut]

            max_unc = min(s_unc.shape[-1], t_unc.shape[-1], uncommon_topk)
            if max_unc > 0:
                # Sort descending and truncate
                if s_unc.shape[-1] > max_unc:
                    s_unc = torch.topk(s_unc, k=max_unc, dim=-1, largest=True).values
                if t_unc.shape[-1] > max_unc:
                    t_unc = torch.topk(t_unc, k=max_unc, dim=-1, largest=True).values

                min_len = min(s_unc.shape[-1], t_unc.shape[-1])
                s_unc = s_unc[..., :min_len]
                t_unc = t_unc[..., :min_len]

                l1_per_pos = F.l1_loss(s_unc, t_unc, reduction='none').sum(dim=-1)  # [B, S]
                l1_uncommon = (l1_per_pos * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        # --- ULD (Unmatched Logit Distillation) ---
        uld_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        if uncommon_s.numel() > 0 and loss_mask.sum() > 0:
            s_unc_probs = student_probs[..., uncommon_s.to(student_logits.device)]
            t_unc_tokens = shift_labels[loss_mask.bool()]

            # For unmatched tokens, compute student-side CE as ULD signal
            s_unc_logits = shift_student_logits[..., uncommon_s.to(student_logits.device)]
            s_unc_logits_flat = s_unc_logits.reshape(-1, s_unc_logits.shape[-1])
            uld_per_token = -(s_unc_probs * (s_unc_probs + eps).log()).sum(dim=-1)
            uld_loss = (uld_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        # Weighted combination
        gamma_l1 = max(0.0, 1.0 - self.gamma_kl - self.gamma_uld)
        kd_loss = (
            self.gamma_kl * common_kl +
            gamma_l1 * l1_uncommon +
            self.gamma_uld * uld_loss
        ) * T * T

        # Top-1 accuracy on common subset
        with torch.no_grad():
            if common_s.numel() > 0:
                s_c = student_log_probs[..., common_s.to(student_logits.device)]
                t_c = teacher_log_probs[..., common_t.to(student_logits.device)]
                s_top1 = s_c.argmax(dim=-1)
                t_top1 = t_c.argmax(dim=-1)
                matches = (s_top1 == t_top1) & loss_mask.bool()
                common_acc = matches.sum().float() / loss_mask.sum().clamp(min=1.0)
            else:
                common_acc = torch.tensor(0.0)

        metrics = {
            'kd_loss': kd_loss.item(),
            'kl_common': common_kl.item(),
            'l1_uncommon': l1_uncommon.item(),
            'uld_loss': uld_loss.item(),
            'proj_accuracy': common_acc.item(),
        }
        return kd_loss, metrics

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_mapping_statistics(self, teacher_index: int = 0) -> Dict:
        """Return statistics about the projection matrix."""
        self._ensure_projection_matrices_built()
        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"No projection matrix for teacher {teacher_index}")

        student_indices = self.projection_student_indices_list[teacher_index]
        values = self.projection_values_list[teacher_index].float()

        nnz = student_indices.numel()
        total_elements = self.student_vocab_size * self.teacher_vocab_sizes[teacher_index]
        sparsity = 1.0 - (nnz / total_elements) if total_elements > 0 else 0.0

        exact_mask = values == 1.0
        exact_students = student_indices[exact_mask].unique()
        exact_matched = exact_students.numel()

        multi_mask = (values > 0) & (values < 1.0)
        multi_students = student_indices[multi_mask].unique()
        multi_students = multi_students[
            ~multi_students.unsqueeze(1).eq(exact_students.unsqueeze(0)).any(dim=1)
        ]
        multi_token_matched = multi_students.numel()

        unmatched = self.student_vocab_size - exact_matched - multi_token_matched

        return {
            'total_student_tokens': self.student_vocab_size,
            'exact_matched': exact_matched,
            'multi_token_matched': multi_token_matched,
            'unmatched': unmatched,
            'sparsity': sparsity,
        }