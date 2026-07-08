import os
from typing import List, Optional

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import CTKDLoss
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

logger = get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', '/nas/disk1/qwen2.5-0.5b-instruct')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', '/model/Qwen3-0.6B')
DATASET_ID = os.environ.get('DATASET_ID', '/model/liujihui/twinkle_client_st/httpserver/models/DG04F8511A00100002/messages.jsonl')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
SHARED_TEACHER_GPUS = bool(os.environ.get('SHARED_TEACHER_GPUS', False))
NUM_GPUS = 2

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 10))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 4))

CTKD_TEMPERATURE = float(os.environ.get('CTKD_TEMPERATURE', 0.8))
CTKD_MAX_LENGTH = int(os.environ.get('CTKD_MAX_LENGTH', 4))
CTKD_BETA = float(os.environ.get('CTKD_BETA', 0.95))
CTKD_GAMMA = float(os.environ.get('CTKD_GAMMA', 0.1))
CTKD_LOSS_TYPE = os.environ.get('CTKD_LOSS_TYPE', 'pkl')
CTKD_TOPK = int(os.environ.get('CTKD_TOPK', 512))

ADAPTER_NAME = 'default'
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 2048))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))
SHARED_TEACHER_GPUS = bool(os.environ.get('SHARED_TEACHER_GPUS', False))


# ── Utility ───────────────────────────────────────────────────────────────────

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[List[Optional[List[tuple]]]],
    topk: int = 64,
) -> dict:
    batch_logprobs = []
    batch_indices = []

    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        for pos_topk in seq_topk:
            if pos_topk is None:
                seq_logprobs.append([0.0] * topk)
                seq_indices.append([0] * topk)
            else:
                seq_logprobs.append([lp for _, lp in pos_topk])
                seq_indices.append([tid for tid, _ in pos_topk])
        batch_logprobs.append(seq_logprobs)
        batch_indices.append(seq_indices)

    max_len = max(len(seq) for seq in batch_logprobs) if batch_logprobs else 1
    for i in range(len(batch_logprobs)):
        pad_len = max_len - len(batch_logprobs[i])
        if pad_len > 0:
            batch_logprobs[i].extend([[0.0] * topk] * pad_len)
            batch_indices[i].extend([[0] * topk] * pad_len)

    # Roll to align with labels (first position has no valid logprobs)
    return {
        'teacher_topk_logprobs': torch.roll(torch.tensor(batch_logprobs, dtype=torch.float32), shifts=-1, dims=1),
        'teacher_topk_indices': torch.roll(torch.tensor(batch_indices, dtype=torch.long), shifts=-1, dims=1),
    }


def align_teacher_logprobs_to_student(
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    student_input_ids_list: List[torch.Tensor],
    student_tokenizer,
    teacher_tokenizer,
    topk: int = 512,
    pad_token_id: int = 0,
) -> dict:
    batch_size = len(student_input_ids_list)
    max_student_len = max(len(ids) for ids in student_input_ids_list)

    aligned_logprobs = torch.zeros(batch_size, max_student_len, topk, dtype=teacher_topk_logprobs.dtype)
    aligned_indices = torch.zeros(batch_size, max_student_len, topk, dtype=teacher_topk_indices.dtype)

    for b in range(batch_size):
        student_ids = student_input_ids_list[b]
        student_len = len(student_ids)

        # Build student character spans by incremental decoding
        student_char_spans = []
        for pos in range(student_len):
            prefix_text = student_tokenizer.decode(student_ids[:pos+1].tolist(), skip_special_tokens=False)
            if pos == 0:
                start_char = 0
            else:
                prev_prefix = student_tokenizer.decode(student_ids[:pos].tolist(), skip_special_tokens=False)
                start_char = len(prev_prefix)
            end_char = len(prefix_text)
            student_char_spans.append((start_char, end_char))

        # Find actual teacher sequence length (before padding)
        teacher_seq_len = teacher_input_ids.shape[1]
        for t in range(teacher_input_ids.shape[1] - 1, -1, -1):
            if teacher_input_ids[b, t].item() != pad_token_id:
                teacher_seq_len = t + 1
                break

        # Build teacher character spans
        teacher_char_spans = []
        teacher_ids_list = teacher_input_ids[b, :teacher_seq_len].tolist()
        for pos in range(teacher_seq_len):
            prefix_text = teacher_tokenizer.decode(teacher_ids_list[:pos+1], skip_special_tokens=False)
            if pos == 0:
                start_char = 0
            else:
                prev_prefix = teacher_tokenizer.decode(teacher_ids_list[:pos], skip_special_tokens=False)
                start_char = len(prev_prefix)
            end_char = len(prefix_text)
            teacher_char_spans.append((start_char, end_char))

        # For each student position, find the best-matching teacher position
        for s_pos in range(student_len):
            s_start, s_end = student_char_spans[s_pos]

            best_t_pos = -1
            best_overlap = 0
            for t_pos in range(teacher_seq_len):
                t_start, t_end = teacher_char_spans[t_pos]
                overlap = min(s_end, t_end) - max(s_start, t_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_t_pos = t_pos

            if best_t_pos >= 0 and best_t_pos < teacher_topk_logprobs.shape[1]:
                aligned_logprobs[b, s_pos] = teacher_topk_logprobs[b, best_t_pos]
                aligned_indices[b, s_pos] = teacher_topk_indices[b, best_t_pos]

    return {
        'teacher_topk_logprobs': aligned_logprobs,
        'teacher_topk_indices': aligned_indices,
    }
def create_dataset():
    """创建用于蒸馏的全文(prompt + response)数据集。

    数据集使用 student tokenizer 编码。Teacher 会将文本解码后，
    使用自己的 tokenizer 重新编码，以实现跨 tokenizer 知识蒸馏。
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH)
    dataset.encode(load_from_cache_file=True)
    return dataset


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    import time
    start_time = time.perf_counter()
    print('记录开始时间')

    # Initialize device groups based on shared mode
    if SHARED_TEACHER_GPUS:
        device_groups = [
            DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
            DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
        ]
    else:
        device_groups = [
            DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
            DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
        ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"代码initialize执行耗时: {elapsed:.6f} 秒")
    start_time = end_time

    # ── Student model (trainable) ──────────────────────────────────────────────
    student_model = TransformersModel(
        model_id=STUDENT_MODEL_ID,
        device_mesh=model_mesh,
        remote_group='student_model',
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"代码student_model: {elapsed:.6f} 秒")
    start_time = end_time

    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules='all-linear',
    )
    student_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    student_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1)

    # ── Configure CTKDLoss ─────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"代码AutoTokenizer: {elapsed:.6f} 秒")
    start_time = end_time

    loss_fn = CTKDLoss(
        student_tokenizer=student_tokenizer,
        teacher_tokenizer_group=[teacher_tokenizer],
        max_length=CTKD_MAX_LENGTH,
        beta=CTKD_BETA,
        gamma=CTKD_GAMMA,
        loss_type=CTKD_LOSS_TYPE,
        temperature=CTKD_TEMPERATURE,
        device=torch.device('npu:0'),
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template('QwenTemplate', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"代码loss_fn: {elapsed:.6f} 秒")
    start_time = end_time

    # Log configuration
    logger.info(f'GPU Configuration: MODEL_GPUS={MODEL_GPUS}, SAMPLER_GPUS={SAMPLER_GPUS}, SHARED_TEACHER_GPUS={SHARED_TEACHER_GPUS}')
    logger.info(f'Total GPUs required: {NUM_GPUS}')

    # Log projection matrix statistics with validation
    stats = loss_fn.get_mapping_statistics()
    logger.info(f'CTKD Projection Matrix Statistics: {stats}')

    # Validate vocabulary coverage
    coverage_ratio = stats['exact_matched'] / stats['total_student_tokens']
    logger.info(f'Vocabulary coverage ratio: {coverage_ratio:.2%}')
    if coverage_ratio < 0.3:
        logger.warning(f"Low vocabulary coverage ({coverage_ratio:.2%}), consider using models with similar tokenizers")
        logger.warning("This may cause poor distillation performance and gradient issues")

    # ── Teacher vLLM samplers ──────────────────────────────────────────────────
    if SHARED_TEACHER_GPUS:
        teacher_sampler = vLLMSampler(
            model_id=TEACHER_MODEL_ID,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler',
            instance_id='teacher_1'
        )
        teacher_sampler.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID)
    else:
        teacher_sampler = vLLMSampler(
            model_id=TEACHER_MODEL_ID,
            engine_args={
                'gpu_memory_utilization': 0.75,
                'max_model_len': 4096,
                'logprobs_mode': 'raw_logprobs',
                'max_logprobs': CTKD_TOPK,
            },
            device_mesh=sampler_mesh,
            remote_group='teacher_sampler',
        )
        teacher_sampler.set_template('QwenTemplate', model_id=TEACHER_MODEL_ID)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset=create_dataset(),
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    # ── Training Loop ──────────────────────────────────────────────────────────
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        if callable(batch):
            batch = batch()

        # ── Step 1: Decode student tokens to text for teacher ──────────────────
        from twinkle.data_format import Trajectory

        teacher_inputs = []
        student_input_ids_list = []
        for item in batch:
            text = student_tokenizer.decode(item['input_ids'], skip_special_tokens=False)
            teacher_inputs.append({'messages': [{'role': 'user', 'content': text}]})
            student_input_ids_list.append(torch.tensor(item['input_ids']))

        # ── Step 2: Teacher computes top-k logprobs ────────────────────────────
        # CRITICAL FIX: Use max_tokens=0 to compute prompt_logprobs only, not generate new tokens
        teacher_response = teacher_sampler.sample(
            teacher_inputs,
            SamplingParams(max_tokens=0, temperature=1.0, prompt_logprobs=CTKD_TOPK),
        )

        # ── Step 3: Convert teacher responses ──────────────────────────────────
        teacher_input_data = [seq.new_input_feature for resp in teacher_response for seq in resp.sequences]

        # ── Step 4: Prepare teacher output with alignment ──────────────────────
        # 4a. Convert topk to tensor format (teacher's sequence length)
        topk_data = convert_topk_prompt_logprobs(
            [resp.topk_prompt_logprobs for resp in teacher_response],
            topk=CTKD_TOPK,
        )

        # 4b. Get teacher input_ids for alignment
        import torch.nn.utils.rnn as rnn_utils
        teacher_input_ids_list = [torch.tensor(item['input_ids']) for item in teacher_input_data]
        teacher_input_ids = rnn_utils.pad_sequence(teacher_input_ids_list, batch_first=True)

        # 4c. CRITICAL FIX: Align teacher logprobs to student token positions
        #     This ensures position-wise KL divergence is computed on semantically
        #     corresponding tokens, not just same index positions.
        aligned_topk = align_teacher_logprobs_to_student(
            teacher_topk_logprobs=topk_data['teacher_topk_logprobs'],
            teacher_topk_indices=topk_data['teacher_topk_indices'],
            teacher_input_ids=teacher_input_ids,
            student_input_ids_list=student_input_ids_list,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            topk=CTKD_TOPK,
        )

        # 4d. Create teacher labels aligned to student sequence length
        #     For CTKD, we want to distill over ALL token positions (not just response).
        #     So we use input_ids as labels (no -100 masking) instead of student's labels
        #     which masks prompt positions with -100.
        student_labels_list = []
        for item in batch:
            # Use input_ids as labels for CTKD - all positions should participate in loss
            # This is different from standard supervised training where prompt is masked
            labels = torch.tensor(item['input_ids'])
            student_labels_list.append(labels)
        student_labels = rnn_utils.pad_sequence(student_labels_list, batch_first=True, padding_value=-100)

        # DEBUG: Log teacher_labels statistics
        print(f'[CTKD DEBUG] mopd_ctkd: student_labels shape: {student_labels.shape}')
        print(f'[CTKD DEBUG] mopd_ctkd: student_labels -100 count: {(student_labels == -100).sum().item()}')
        print(f'[CTKD DEBUG] mopd_ctkd: student_labels non -100 count: {(student_labels != -100).sum().item()}')
        print(f'[CTKD DEBUG] mopd_ctkd: student_labels first sample first 20 values: {student_labels[0, :20].tolist()}')

        # DEBUG: Log teacher topk data statistics
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_topk_logprobs shape: {aligned_topk["teacher_topk_logprobs"].shape}')
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_topk_logprobs range: [{aligned_topk["teacher_topk_logprobs"].min().item():.4f}, {aligned_topk["teacher_topk_logprobs"].max().item():.4f}]')
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_topk_indices shape: {aligned_topk["teacher_topk_indices"].shape}')
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_topk_indices range: [{aligned_topk["teacher_topk_indices"].min().item()}, {aligned_topk["teacher_topk_indices"].max().item()}]')
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_input_ids shape: {teacher_input_ids.shape}')
        print(f'[CTKD DEBUG] mopd_ctkd: teacher_input_ids first sample first 20 values: {teacher_input_ids[0, :20].tolist()}')

        # Create teacher_output dict with aligned data
        teacher_output = {
            'teacher_labels': [student_labels],  # All positions participate in CTKD loss
            'teacher_input_ids': [teacher_input_ids],
            'teacher_topk_logprobs_group': [aligned_topk['teacher_topk_logprobs']],
            'teacher_topk_indices_group': [aligned_topk['teacher_topk_indices']],
        }

        # ── Step 5: Student forward + CTKD backward ────────────────────────────
        student_model.forward_backward(
            inputs=batch,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )

        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # 5. Logging
        if optim_step > 0 and optim_step % 2 == 0:
            metric = student_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # ── Checkpoint ─────────────────────────────────────────────────────────
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(f'mopd-ctkd-ckpt-{optim_step}', adapter_name=ADAPTER_NAME)

        optim_step += 1

    # Save final checkpoint
    student_model.save('mopd-ctkd-final', adapter_name=ADAPTER_NAME)
    logger.info('MOPD CTKD training completed.')


if __name__ == '__main__':
    train()