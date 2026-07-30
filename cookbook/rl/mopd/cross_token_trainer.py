"""Cross-Token Knowledge Distillation Training (Off-Policy).

This script implements the NeMo-style X-Token training pipeline using
CrossTokenLoss, supporting both P-KL and H-KL loss types for cross-tokenizer
knowledge distillation. This is the twinkle equivalent of NeMo's
``xtoken_off_policy_distillation.py``.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)

Off-Policy Pipeline:
    1. Dataloader provides pre-tokenized (prompt + response) batches using
       the student tokenizer.
    2. Each batch item is decoded to text and re-encoded with the teacher
       tokenizer for cross-tokenizer teacher input.
    3. Teacher TransformersModel computes full logits on teacher-tokenized
       sequences.
    4. Student TransformersModel runs forward_backward() with CrossTokenLoss
       using teacher full logits.

Architecture (Ray):
    +-----------------------------------------------------------------+
    | Driver (CPU)                                                    |
    |  dataloader -> decode student tokens -> re-encode for teacher   |
    |  teacher.forward_only(return_logits=True) -> teacher logits      |
    |  student.forward_backward(**teacher_output) -> Loss             |
    +-----------------------------------------------------------------+
          |                    |                       |
     DataLoader      TransformersModel         TransformersModel
                     (teacher sampler group)   (student model group)

Environment variables (all optional):
    STUDENT_MODEL_ID          – Student model path (default: /model/Qwen3-0.6B)
    TEACHER_MODEL_ID          – Teacher model path (default: /nas/disk1/Qwen3-1.7B)
    DATASET_ID                – Dataset path (JSONL file)
    MODEL_GPUS                – GPUs for student model (default: 1)
    SAMPLER_GPUS              – GPUs for teacher model / sampler (default: 1)
    BATCH_SIZE                – Global batch size (default: 2)
    MAX_STEPS                 – Total optimisation steps (default: 5)
    LR                        – Learning rate (default: 1e-5)
    GRADIENT_ACCUMULATION_STEPS – Gradient accumulation (default: 4)
    LOSS_TYPE                 – 'pkl' or 'hkl' (default: 'pkl')
    TEMPERATURE               – Distillation temperature (default: 0.8)
    MAX_LENGTH                – Max span length for multi-token matching (default: 4)
    BETA                      – Base weight for projection (default: 0.95)
    GAMMA                     – Decay rate for multi-token weights (default: 0.1)
    GAMMA_KL                  – Weight for common-KL in H-KL (default: 1.0)
    GAMMA_ULD                 – Weight for ULD in H-KL (default: 0.5)
    VOCAB_TOPK                – Top-k vocab for P-KL subset (default: 64)
    UNCOMMON_TOPK             – Top-k for uncommon L1 in H-KL (default: 8192)
    MAX_LENGTH_SEQ            – Max sequence length for dataset (default: 2048)
"""

import os
from typing import List, Optional

import torch
import torch.nn.utils.rnn as rnn_utils
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
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
TEACHER_GPUS = int(os.environ.get('TEACHER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + TEACHER_GPUS

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
KL_LOSS_WEIGHT = float(os.environ.get('KL_LOSS_WEIGHT', 1.0))
CE_LOSS_WEIGHT = float(os.environ.get('CE_LOSS_WEIGHT', 1.0))
DYNAMIC_LOSS_SCALING = os.environ.get('DYNAMIC_LOSS_SCALING', 'false').lower() in ('true', '1', 'yes')

ADAPTER_NAME = 'default'


# ── Dataset ───────────────────────────────────────────────────────────────────

def create_dataset():
    """Create a full-text (prompt + response) dataset for off-policy distillation.

    The dataset is encoded with the student tokenizer; teacher inputs are
    produced by decoding to text and re-encoding with the teacher tokenizer
    at each step.
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH_SEQ)
    dataset.encode(load_from_cache_file=True)
    return dataset


# ── Utility ───────────────────────────────────────────────────────────────────

def prepare_teacher_inputs(
    batch: list,
    student_tokenizer,
    teacher_tokenizer,
) -> list:
    """Decode student tokens to text and re-encode with teacher tokenizer.

    For off-policy distillation with different tokenizers, the teacher needs
    its own tokenization of the same text content. We decode student input_ids
    to raw text, then encode with the teacher tokenizer.

    Args:
        batch: List of dicts with 'input_ids' (list of token IDs).
        student_tokenizer: Student model tokenizer (for decoding).
        teacher_tokenizer: Teacher model tokenizer (for re-encoding).

    Returns:
        List of new_input_feature dicts with teacher-tokenized 'input_ids'.
    """
    teacher_inputs = []
    for item in batch:
        student_ids = item['input_ids']
        text = student_tokenizer.decode(student_ids, skip_special_tokens=False)

        # Re-encode with teacher tokenizer
        teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=True)
        # Truncate to max sequence length
        teacher_ids = teacher_ids[:MAX_LENGTH_SEQ]

        teacher_inputs.append({
            'input_ids': teacher_ids,
            'labels': teacher_ids[:],  # Use same ids for CE masking
        })
    return teacher_inputs


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    """Main training loop for off-policy cross-tokenizer distillation."""
    import time
    start_time = time.perf_counter()
    print('Recording start time')

    # ── Initialize device groups ──────────────────────────────────────────────
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
        DeviceGroup(name='teacher', ranks=TEACHER_GPUS, device_type='npu'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    teacher_mesh = DeviceMesh.from_sizes(
        world_size=TEACHER_GPUS, dp_size=TEACHER_GPUS
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

    # LoRA configuration for efficient fine-tuning
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
        STUDENT_MODEL_ID, trust_remote_code=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL_ID, trust_remote_code=True
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
    )
    elapsed = time.perf_counter() - start_time
    print(f"loss_fn init: {elapsed:.6f}s")
    start_time = time.perf_counter()

    # Log configuration
    logger.info(
        f'GPU Configuration: MODEL_GPUS={MODEL_GPUS}, '
        f'TEACHER_GPUS={TEACHER_GPUS}'
    )
    logger.info(f'Total GPUs required: {NUM_GPUS} (student + teacher)')

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

    # ── Teacher TransformersModel ─────────────────────────────────────────────
    # Use TransformersModel to get full-vocab logits from the teacher.
    # vLLMSampler is limited to prompt_logprobs top-k (typically <= 64).
    teacher_model = TransformersModel(
        model_id=TEACHER_MODEL_ID,
        device_mesh=teacher_mesh,
        remote_group='teacher',
    )
    teacher_model.set_template('Template', model_id=TEACHER_MODEL_ID)

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
        f'CrossToken Off-Policy Training | '
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
        f'max_steps={MAX_STEPS}  kl_w={KL_LOSS_WEIGHT}  ce_w={CE_LOSS_WEIGHT}'
    )
    if DYNAMIC_LOSS_SCALING:
        logger.info('  dynamic_loss_scaling=enabled')

    # ── Training Loop (Off-Policy) ────────────────────────────────────────────
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        if callable(batch):
            batch = batch()

        # ── Step 1: Prepare teacher inputs via decode-re-encode ───────────────
        teacher_inputs = prepare_teacher_inputs(
            batch, student_tokenizer, teacher_tokenizer
        )

        # ── Step 2: Teacher forward → full logits ────────────────────────────
        teacher_outputs = teacher_model.forward_only(
            inputs=teacher_inputs,
            return_logits=True,
            temperature=1.0,
            disable_lora=True,
            adapter_name='',
        )
        teacher_outputs = teacher_outputs()  # remote_function → actual result

        # ── Step 3: Package teacher output ────────────────────────────────────
        teacher_logits = teacher_outputs['logits']

        # Build teacher input_ids tensor (padded) for optional alignment
        teacher_prompt_ids = [
            torch.tensor(feat['input_ids'], dtype=torch.long)
            for feat in teacher_inputs
        ]
        teacher_input_ids = rnn_utils.pad_sequence(
            teacher_prompt_ids, batch_first=True, padding_value=0,
        )

        # For cross-tokenizer training, use student input_ids as labels
        # so that all positions participate in distillation (not just response).
        # The -100 mask from standard supervised training masks prompt positions
        # but for KD we want all positions.
        student_labels_list = []
        for item in batch:
            labels = torch.tensor(item['input_ids'], dtype=torch.long)
            student_labels_list.append(labels)
        student_labels = rnn_utils.pad_sequence(
            student_labels_list, batch_first=True, padding_value=-100,
        )

        teacher_output = {
            'teacher_logits_group': [teacher_logits],
            'teacher_input_ids_group': [teacher_input_ids],
            'teacher_labels': [student_labels],
        }

        # DEBUG: Log shapes
        print(f"\n[Step {optim_step}] Data shapes:")
        print(f"  teacher_logits: {teacher_logits.shape}")
        print(f"  student_labels: {student_labels.shape}")
        print(
            f"  student_labels non-(-100): "
            f"{(student_labels != -100).sum().item()}"
        )

        # ── Step 4: Student forward + CrossToken backward ────────────────────
        student_model.forward_backward(
            inputs=batch,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
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
                f'cross-token-ckpt-{optim_step}', adapter_name=ADAPTER_NAME,
            )

        optim_step += 1

    # ── Save final checkpoint ─────────────────────────────────────────────────
    student_model.save('cross-token-final', adapter_name=ADAPTER_NAME)
    logger.info(
        f'CrossToken off-policy training completed after {optim_step} steps.'
    )


if __name__ == '__main__':
    train()