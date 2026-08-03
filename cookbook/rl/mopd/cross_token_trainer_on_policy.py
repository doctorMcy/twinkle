"""Cross-Token Knowledge Distillation Training (On-Policy).

This script implements the on-policy X-Token training pipeline using
CrossTokenLoss. The student model generates completions on-the-fly,
and the teacher distills its knowledge on the student's own output
distribution — matching the on-policy distillation paradigm.

This is the on-policy counterpart of ``cross_token_trainer.py`` (off-policy).

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)
    "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
    (https://arxiv.org/abs/2306.13649)

On-Policy Pipeline:
    1. Sync student model weights to student vLLM sampler.
    2. Student vLLM sampler generates completions on-the-fly.
    3. Decode student-generated text, re-encode with teacher tokenizer.
    4. Teacher TransformersModel computes full logits.
    5. Student TransformersModel runs forward_backward() with CrossTokenLoss
       on its own generated data.

Architecture (Ray):
    +-----------------------------------------------------------------+
    | Driver (CPU)                                                    |
    |  ckpt_manager.sync_weights() --> sync LoRA to student sampler   |
    |  student_vllm.sample() --> on-policy completions                |
    |  decode-re-encode for teacher tokenizer                         |
    |  teacher.forward_only() --> teacher full logits                 |
    |  student.forward_backward(teacher_output=...) --> Loss          |
    +-----------------------------------------------------------------+
          |               |                    |
     DataLoader      vLLMSampler         TransformersModel x2
                   (student gen)       (teacher + student)

Environment variables (all optional):
    STUDENT_MODEL_ID          – Student model path
    TEACHER_MODEL_ID          – Teacher model path
    DATASET_ID                – Prompt-only dataset (JSONL file)
    MODEL_GPUS                – GPUs for student model (default: 1)
    STUDENT_SAMPLER_GPUS      – GPUs for student vLLM sampler (default: 1)
    TEACHER_GPUS              – GPUs for teacher model (default: 1)
    BATCH_SIZE                – Global batch size (default: 2)
    MAX_STEPS                 – Total optimisation steps (default: 5)
    LR                        – Learning rate (default: 1e-5)
    GRADIENT_ACCUMULATION_STEPS – Gradient accumulation (default: 4)
    LOSS_TYPE                 – 'pkl' or 'hkl' (default: 'pkl')
    TEMPERATURE               – Distillation temperature (default: 0.8)
    MAX_LENGTH                – Max span length for multi-token (default: 4)
    BETA                      – Base weight for projection (default: 0.95)
    GAMMA                     – Decay rate for multi-token weights (default: 0.1)
    GAMMA_KL                  – Weight for common-KL in H-KL (default: 1.0)
    GAMMA_ULD                 – Weight for ULD in H-KL (default: 0.5)
    VOCAB_TOPK                – Top-k vocab for P-KL subset (default: 64)
    UNCOMMON_TOPK             – Top-k for uncommon L1 in H-KL (default: 8192)
    MAX_LENGTH_SEQ            – Max sequence length for dataset (default: 2048)
    MAX_NEW_TOKENS            – Max tokens for student generation (default: 2048)
    N_SAMPLES                 – Number of samples per prompt (default: 1)
"""

import os
from typing import List, Optional

import torch
import torch.nn.utils.rnn as rnn_utils
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CrossTokenLoss
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', '/model/Qwen3-0.6B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', '/nas/disk1/Qwen3-1.7B')
DATASET_ID = os.environ.get(
    'DATASET_ID', '/root/twinkle/cookbook/rl/mopd/data.jsonl'
)

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
STUDENT_SAMPLER_GPUS = int(os.environ.get('STUDENT_SAMPLER_GPUS', 1))
TEACHER_GPUS = int(os.environ.get('TEACHER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + STUDENT_SAMPLER_GPUS + TEACHER_GPUS

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 5))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 4))

LOSS_TYPE = os.environ.get('LOSS_TYPE', 'pkl')
TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.8))
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 4))
BETA = float(os.environ.get('BETA', 0.95))
GAMMA = float(os.environ.get('GAMMA', 0.1))
GAMMA_KL = float(os.environ.get('GAMMA_KL', 1.0))
GAMMA_ULD = float(os.environ.get('GAMMA_ULD', 0.5))
VOCAB_TOPK = int(os.environ.get('VOCAB_TOPK', 64))
UNCOMMON_TOPK = int(os.environ.get('UNCOMMON_TOPK', 8192))
MAX_LENGTH_SEQ = int(os.environ.get('MAX_LENGTH_SEQ', 2048))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))
KL_LOSS_WEIGHT = float(os.environ.get('KL_LOSS_WEIGHT', 1.0))
CE_LOSS_WEIGHT = float(os.environ.get('CE_LOSS_WEIGHT', 1.0))
DYNAMIC_LOSS_SCALING = os.environ.get(
    'DYNAMIC_LOSS_SCALING', 'false'
).lower() in ('true', '1', 'yes')

ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Create a prompt-only dataset for on-policy distillation.

    The dataset only contains prompts; the student model generates completions
    on-the-fly. The teacher model computes logprobs on student-generated
    sequences.
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template(
        'Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH_SEQ,
        enable_thinking=False,
    )
    dataset.encode(load_from_cache_file=True)
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def prepare_teacher_inputs_from_student_gen(
    sample_response: list,
    student_tokenizer,
    teacher_tokenizer,
) -> list:
    """Decode student-generated sequences and re-encode for teacher.

    Args:
        sample_response: List of vLLM response objects from student_sampler.sample().
        student_tokenizer: Student model tokenizer (for decoding).
        teacher_tokenizer: Teacher model tokenizer (for re-encoding).

    Returns:
        List of new_input_feature dicts with teacher-tokenized 'input_ids'.
    """
    teacher_inputs = []
    for resp in sample_response:
        for seq in resp.sequences:
            full_ids = seq.new_input_feature['input_ids']
            text = student_tokenizer.decode(full_ids, skip_special_tokens=False)

            teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=True)
            teacher_ids = teacher_ids[:MAX_LENGTH_SEQ]

            teacher_inputs.append({
                'input_ids': teacher_ids,
                'labels': teacher_ids[:],
            })
    return teacher_inputs


def _as_list(ids):
    """Convert token ids (tensor / list / array) to a plain Python list."""
    if isinstance(ids, torch.Tensor):
        return ids.tolist()
    return list(ids)


def print_full_outputs(
    optim_step: int,
    sample_response: list,
    teacher_inputs: list,
    student_tokenizer,
    teacher_tokenizer,
) -> None:
    """Print complete student/teacher outputs (source text + token ids).

    Args:
        optim_step: Current optimisation step (for the log header).
        sample_response: vLLM responses from student_sampler.sample().
        teacher_inputs: Teacher-tokenized inputs from
            prepare_teacher_inputs_from_student_gen() (flat over samples).
        student_tokenizer: Student tokenizer (for decoding).
        teacher_tokenizer: Teacher tokenizer (for decoding).
    """
    print(f"\n{'='*80}")
    print(f"[Step {optim_step}] FULL OUTPUTS (STUDENT → TEACHER):")
    print(f"{'='*80}")
    t_idx = 0
    for i, resp in enumerate(sample_response):
        for j, seq in enumerate(resp.sequences):
            s_ids = seq.new_input_feature['input_ids']
            s_text = student_tokenizer.decode(s_ids, skip_special_tokens=False)
            t_ids = teacher_inputs[t_idx]['input_ids']
            t_text = teacher_tokenizer.decode(t_ids, skip_special_tokens=False)
            t_idx += 1
            print(f"\n  Sample[{i}].Seq[{j}] STUDENT:")
            print(f"    源文本: {s_text}")
            print(f"    tokenid: {_as_list(s_ids)}")
            print(f"  Sample[{i}].Seq[{j}] TEACHER:")
            print(f"    源文本: {t_text}")
            print(f"    tokenid: {_as_list(t_ids)}")


def print_cross_token_mapping(
    optim_step: int,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn,
) -> None:
    """Print per-token student↔teacher token-id mapping with probabilities.

    The probability columns replicate ``CrossTokenLoss._compute_pkl`` exactly
    (same temperature-scaled softmax, same probs/logits heuristic, same
    character-span chunk alignment), so the printed values match the loss
    internals.

    Row format:
        学生投影前 → 学生投影后 → 教师概率 → 教师对学生tokenid的概率
        | 学生tokenId(学生文本) → 教师tokenId(教师文本) [匹配关系]

    Column semantics:
      - 学生投影前: student prob of its own generated token (softmax / T)
      - 学生投影后: projected student prob (on teacher vocab) at the teacher
        token the student token maps to; '-' when unmapped
      - 教师概率: teacher prob of its own token at the aligned teacher position
      - 教师对学生tokenid的概率: teacher prob at the aligned position of the raw
        student token id used directly as a teacher-vocab index; '-' when the
        id is out of the teacher vocab range
      - 匹配关系: 精确 (weight 1.0) / 多token (β·γ^i) / 未匹配
    """
    T = loss_fn.temperature
    eps = 1e-10
    teacher_index = 0
    tkr_vocab_size = loss_fn.teacher_vocab_sizes[teacher_index]
    student_tok = loss_fn.student_tokenizer
    teacher_tok = loss_fn.teacher_tokenizer_group[teacher_index]

    # Projection matrix → {student token id: [(teacher id, weight), ...]}
    s_idx = loss_fn.projection_student_indices_list[teacher_index].cpu()
    t_idx = loss_fn.projection_teacher_indices_list[teacher_index].cpu()
    values = loss_fn.projection_values_list[teacher_index].cpu().float()
    proj_map = {}
    for s, t, v in zip(s_idx.tolist(), t_idx.tolist(), values.tolist()):
        proj_map.setdefault(s, []).append((t, v))

    # ── Shift / vocab trim / chunk alignment (same as _compute_pkl) ─────
    shift_labels = labels[..., 1:]
    s_clen = min(shift_labels.shape[1], student_logits.shape[1] - 1)
    shift_labels = shift_labels[:, :s_clen]

    shift_teacher = teacher_logits[..., :-1, :].float().cpu()
    if shift_teacher.shape[-1] > tkr_vocab_size:
        shift_teacher = shift_teacher[..., :tkr_vocab_size]
    t_clen = shift_teacher.shape[1]

    _, teacher_chunk_id, _ = loss_fn._build_chunk_ids(
        student_ids, teacher_ids, teacher_index,
    )
    teacher_chunk_id = teacher_chunk_id[:, :t_clen]

    # Teacher probs (same probs/logits heuristic as the loss)
    if shift_teacher.max() <= 1.0 and shift_teacher.sum(dim=-1).min() > 0.5:
        teacher_logits_approx = torch.log(shift_teacher.clamp(min=eps))
    else:
        teacher_logits_approx = shift_teacher

    batch_size = shift_labels.shape[0]
    for b in range(batch_size):
        # Student probs + projection (per-sample to bound peak memory)
        shift_student = student_logits[b:b + 1, :-1, :].float().cpu()
        student_probs = torch.softmax(shift_student / T, dim=-1)
        projected = loss_fn._project_student_probs(
            student_probs, teacher_index, tkr_vocab_size,
        )
        teacher_probs = torch.softmax(
            teacher_logits_approx[b:b + 1] / T, dim=-1,
        )

        print(f"\n[Step {optim_step}] CROSS-TOKEN TOKEN MAPPING (sample {b}):")
        print(
            '  学生投影前 → 学生投影后 → 教师概率 → 教师对学生tokenid的概率'
            ' | 学生tokenId(学生文本) → 教师tokenId(教师文本) [匹配关系]'
        )
        for i in range(s_clen):
            s_token = int(shift_labels[b, i])
            if s_token == -100:
                continue

            s_text = student_tok.decode([s_token], skip_special_tokens=False)
            p_before = float(student_probs[0, i, s_token].item())

            # Projection match type: 精确 / 多token / 未匹配
            entries = proj_map.get(s_token, [])
            exact = [e for e in entries if e[1] == 1.0]
            multi = [e for e in entries if 0.0 < e[1] < 1.0]
            if exact:
                t_star, _ = exact[0]
                match_type = '精确'
                p_after = float(projected[0, i, t_star].item())
            elif multi:
                t_star, _ = multi[0]  # first mapping = highest weight (β)
                match_type = '多token'
                p_after = float(projected[0, i, t_star].item())
            else:
                t_star = None
                match_type = '未匹配'
                p_after = None

            # Aligned teacher position: first teacher position of chunk i
            t_positions = (teacher_chunk_id[b] == i).nonzero(as_tuple=True)[0]
            if t_positions.numel() > 0:
                t_pos = int(t_positions[0].item())
                t_own = int(teacher_ids[b, t_pos + 1])
                p_teacher_own = float(teacher_probs[0, t_pos, t_own].item())
                if s_token < tkr_vocab_size:
                    p_teacher_student = float(
                        teacher_probs[0, t_pos, s_token].item()
                    )
                else:
                    p_teacher_student = None
            else:
                p_teacher_own = None
                p_teacher_student = None

            p_after_s = f'{p_after:.4f}' if p_after is not None else '-'
            p_own_s = f'{p_teacher_own:.4f}' if p_teacher_own is not None else '-'
            p_stu_s = (
                f'{p_teacher_student:.4f}'
                if p_teacher_student is not None else '-'
            )
            if t_star is not None:
                t_star_text = teacher_tok.decode(
                    [t_star], skip_special_tokens=False
                )
                t_pair = f'{t_star}({t_star_text})'
            else:
                t_pair = '-'
            print(
                f'  {p_before:.4f} → {p_after_s} → {p_own_s} → {p_stu_s}'
                f' | {s_token}({s_text}) → {t_pair} [{match_type}]'
            )


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training loop for on-policy cross-tokenizer distillation."""
    import time
    start_time = time.perf_counter()
    print('Recording start time')

    # ── Initialize device groups ──────────────────────────────────────────────
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
        DeviceGroup(
            name='student_sampler', ranks=STUDENT_SAMPLER_GPUS, device_type='npu',
        ),
        DeviceGroup(name='teacher', ranks=TEACHER_GPUS, device_type='npu'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(
        world_size=STUDENT_SAMPLER_GPUS, dp_size=STUDENT_SAMPLER_GPUS,
    )
    teacher_mesh = DeviceMesh.from_sizes(
        world_size=TEACHER_GPUS, dp_size=TEACHER_GPUS,
    )

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )
    elapsed = time.perf_counter() - start_time
    print(f"initialize elapsed: {elapsed:.6f}s")
    start_time = time.perf_counter()

    # ── Student model (trainable) ─────────────────────────────────────────────
    student_model = TransformersModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='student_model',
    )
    elapsed = time.perf_counter() - start_time
    print(f"student_model init: {elapsed:.6f}s")
    start_time = time.perf_counter()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
    )
    student_model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    student_model.set_lr_scheduler(
        'CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1,
    )

    # ── Configure CrossTokenLoss ──────────────────────────────────────────────
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(
        STUDENT_MODEL_ID, trust_remote_code=True,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL_ID, trust_remote_code=True,
    )
    elapsed = time.perf_counter() - start_time
    print(f"tokenizer init: {elapsed:.6f}s")
    start_time = time.perf_counter()

    loss_fn = CrossTokenLoss(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer_group=[teacher_tokenizer],
        max_length=MAX_LENGTH,
        beta=BETA,
        gamma=GAMMA,
        loss_type=LOSS_TYPE,
        temperature=TEMPERATURE,
        gamma_kl=GAMMA_KL,
        gamma_uld=GAMMA_ULD,
        vocab_topk=VOCAB_TOPK,
        uncommon_topk=UNCOMMON_TOPK,
        kl_loss_weight=KL_LOSS_WEIGHT,
        ce_loss_weight=CE_LOSS_WEIGHT,
        dynamic_loss_scaling=DYNAMIC_LOSS_SCALING,
        device=torch.device('npu:0'),
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template(
        'Template', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME,
        enable_thinking=False,
    )
    elapsed = time.perf_counter() - start_time
    print(f"loss_fn init: {elapsed:.6f}s")
    start_time = time.perf_counter()

    # Log projection matrix statistics
    stats = loss_fn.get_mapping_statistics()
    logger.info(f'CrossToken Projection Matrix Statistics: {stats}')

    coverage_ratio = stats['exact_matched'] / stats['total_student_tokens']
    logger.info(f'Vocabulary coverage ratio: {coverage_ratio:.2%}')
    if coverage_ratio < 0.3:
        logger.warning(
            f"Low vocabulary coverage ({coverage_ratio:.2%}), "
            "consider using models with similar tokenizers"
        )

    if LOSS_TYPE == 'hkl':
        logger.info(f'H-KL mode: gamma_kl={GAMMA_KL}, gamma_uld={GAMMA_ULD}')
        logger.info(
            f'  Unmatched tokens (ULD): '
            f'{stats["unmatched"]}/{stats["total_student_tokens"]}'
        )

    # ── Student vLLM sampler (for on-policy generation) ───────────────────────
    student_sampler = vLLMSampler(
        model_id=STUDENT_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.75,
            'max_model_len': 4096,
            'enable_lora': True,
            'max_lora_rank': 8,
        },
        device_mesh=sampler_mesh,
        remote_group='student_sampler',
    )
    student_sampler.set_template(
        'Template', model_id=STUDENT_MODEL_ID, enable_thinking=False,
    )

    # ── Teacher TransformersModel (for full logits) ───────────────────────────
    teacher_model = TransformersModel(
        model_id=TEACHER_MODEL_ID,
        device_mesh=teacher_mesh,
        remote_group='teacher',
    )
    teacher_model.set_template(
        'Template', model_id=TEACHER_MODEL_ID, enable_thinking=False,
    )

    # ── Checkpoint manager for weight sync ────────────────────────────────────
    ckpt_manager = CheckpointEngineManager(
        model=student_model, sampler=student_sampler,
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dataset(),
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    logger.info(get_device_placement())
    logger.info(
        f'CrossToken On-Policy Training | '
        f'student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}'
    )
    logger.info(
        f'  loss_type={LOSS_TYPE}  T={TEMPERATURE}  vocab_topk={VOCAB_TOPK}'
    )
    logger.info(
        f'  beta={BETA}  gamma={GAMMA}  max_length={MAX_LENGTH}'
    )
    if LOSS_TYPE == 'hkl':
        logger.info(
            f'  gamma_kl={GAMMA_KL}  gamma_uld={GAMMA_ULD}  '
            f'uncommon_topk={UNCOMMON_TOPK}'
        )
    logger.info(
        f'  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}  '
        f'max_steps={MAX_STEPS}  max_new_tokens={MAX_NEW_TOKENS}  '
        f'n_samples={N_SAMPLES}'
    )
    if DYNAMIC_LOSS_SCALING:
        logger.info('  dynamic_loss_scaling=enabled')

    # ── Training Loop (On-Policy) ─────────────────────────────────────────────
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        if callable(batch):
            batch = batch()

        # ── Step 1: Sync student weights to sampler ───────────────────────────
        ckpt_manager.sync_weights(merge_and_sync=False)
        student_sampler.reset_prefix_cache()
        student_sampler.reset_encoder_cache()

        # ── Step 2: Student generates completions ────────────────────────────
        sample_response = student_sampler.sample(
            batch,
            SamplingParams(
                max_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                num_samples=N_SAMPLES,
            ),
        )

        # Extract generated sequences (prompt + student-generated response)
        input_data = [
            seq.new_input_feature
            for resp in sample_response
            for seq in resp.sequences
        ]

        # ── Step 3: Prepare teacher inputs (decode + re-encode) ──────────────
        teacher_inputs = prepare_teacher_inputs_from_student_gen(
            sample_response, student_tokenizer, teacher_tokenizer,
        )

        # DEBUG: 完整输出(源文本 + tokenid)
        print_full_outputs(
            optim_step, sample_response, teacher_inputs,
            student_tokenizer, teacher_tokenizer,
        )

        # ── Step 4: Teacher forward → full logits ────────────────────────────
        teacher_outputs = teacher_model.forward_only(
            inputs=teacher_inputs,
            return_logits=True,
            temperature=1.0,
            disable_lora=True,
            adapter_name='',
        )
        teacher_outputs = teacher_outputs()  # remote_function → actual result

        # ── Step 5: Package teacher output ───────────────────────────────────
        teacher_logits = teacher_outputs['logits']

        teacher_prompt_ids = [
            torch.tensor(feat['input_ids'], dtype=torch.long)
            for feat in teacher_inputs
        ]
        teacher_input_ids = rnn_utils.pad_sequence(
            teacher_prompt_ids, batch_first=True, padding_value=0,
        )

        # Use student-generated input_ids as labels (all positions participate
        # in distillation, not just response tokens).
        student_labels_list = [
            torch.tensor(feat['input_ids'], dtype=torch.long)
            for feat in input_data
        ]
        student_labels = rnn_utils.pad_sequence(
            student_labels_list, batch_first=True, padding_value=-100,
        )
        student_ids = rnn_utils.pad_sequence(
            student_labels_list, batch_first=True, padding_value=0,
        )

        teacher_output = {
            'teacher_logits_group': [teacher_logits],
            'teacher_input_ids_group': [teacher_input_ids],
            'teacher_labels': [student_labels],
        }

        # DEBUG: Log shapes
        print(f"\n[Step {optim_step}] Data shapes:")
        print(f"  input_data samples: {len(input_data)}")
        print(f"  teacher_logits: {teacher_logits.shape}")
        print(f"  student_labels: {student_labels.shape}")
        print(
            f"  student_labels non-(-100): "
            f"{(student_labels != -100).sum().item()}"
        )

        # ── Step 6: Student forward + CrossToken backward ────────────────────
        student_outputs = student_model.forward_backward(
            inputs=input_data,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )
        student_outputs = student_outputs()  # remote_function → actual result

        # DEBUG: 逐 token 映射关系(学生/教师 tokenid + 概率)
        print_cross_token_mapping(
            optim_step,
            student_outputs['logits'],
            teacher_logits,
            student_ids,
            teacher_input_ids,
            student_labels,
            loss_fn,
        )

        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # ── Logging ───────────────────────────────────────────────────────────
        if optim_step > 0 and optim_step % 2 == 0:
            metric = student_model.calculate_metric(
                is_training=True, adapter_name=ADAPTER_NAME,
            )
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # ── Checkpoint ────────────────────────────────────────────────────────
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(
                f'cross-token-on-policy-ckpt-{optim_step}',
                adapter_name=ADAPTER_NAME,
            )

        optim_step += 1

    # ── Save final checkpoint ─────────────────────────────────────────────────
    student_model.save(
        'cross-token-on-policy-final', adapter_name=ADAPTER_NAME,
    )
    logger.info(
        f'CrossToken on-policy training completed after {optim_step} steps.'
    )


if __name__ == '__main__':
    train()