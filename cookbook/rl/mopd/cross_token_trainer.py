"""Cross-Token Knowledge Distillation Training (On-Policy).

This script implements the X-Token training pipeline using CrossTokenLoss,
which supports both P-KL and H-KL (with ULD) loss types for cross-tokenizer
knowledge distillation.

Reference: https://arxiv.org/pdf/2605.21699

On-Policy Pipeline:
    1. Sync student model weights to student vLLM sampler.
    2. Student vLLM sampler generates completions on-the-fly.
    3. Teacher vLLM sampler computes top-k prompt logprobs on student-generated sequences.
    4. Student TransformersModel runs forward_backward() with CrossTokenLoss.

Architecture (Ray):
    +-----------------------------------------------------------------+
    | Driver (CPU)                                                    |
    |  ckpt_manager.sync_weights() --> sync LoRA to student sampler  |
    |  student_vllm.sample() --> on-policy completions            |
    |  teacher_vllm.sample(prompt_logprobs=k) --> teacher lps        |
    |  student_model.forward_backward(teacher_output=...) --> Loss   |
    +-----------------------------------------------------------------+
         |               |                    |
    DataLoader      vLLMSampler x2     TransformersModel
                  student + teacher      (student)

Environment variables (all optional):
    STUDENT_MODEL_ID  – (default: ms://Qwen/Qwen3-0.6B)
    TEACHER_MODEL_ID  – (default: ms://Qwen/Qwen2.5-7B-Instruct)
    DATASET_ID        – (default: ms://AI-ModelScope/shareAI-Llama3-DPO-zh-en-emoji)
    MODEL_GPUS        – GPUs for student model               (default: 1)
    SAMPLER_GPUS      – GPUs for each vLLM sampler           (default: 1)
    BATCH_SIZE        – global batch size                    (default: 8)
    MAX_STEPS         – total optimisation steps             (default: 4)
    LR                – learning rate                        (default: 1e-5)
    LOSS_TYPE         – loss type: 'pkl' or 'hkl'            (default: 'pkl')
    TEMPERATURE       – distillation temperature             (default: 0.8)
    MAX_LENGTH        – max span length for multi-token      (default: 4)
    BETA              – base weight for projection           (default: 0.95)
    GAMMA             – decay rate for multi-token weights   (default: 0.1)
    GAMMA_KL          – weight for common-KL in H-KL         (default: 1.0)
    GAMMA_ULD         – weight for ULD loss in H-KL         (default: 0.5)
    TOPK              – top-k vocab for teacher logprobs     (default: 512)
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

# -- Configuration --
STUDENT_MODEL_ID = os.environ.get('STUDENT_MODEL_ID', '/model/Qwen3-0.6B')
TEACHER_MODEL_ID = os.environ.get('TEACHER_MODEL_ID', '/nas/disk1/Qwen3-1.7B')
DATASET_ID = os.environ.get('DATASET_ID', '/root/twinkle/cookbook/rl/mopd/data.jsonl')

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 1))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
NUM_GPUS = MODEL_GPUS + 2 * SAMPLER_GPUS  # student_model + student_sampler + teacher_sampler

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
TOPK = int(os.environ.get('TOPK', 64))

ADAPTER_NAME = 'default'
MAX_LENGTH_SEQ = int(os.environ.get('MAX_LENGTH_SEQ', 2048))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 2048))
N_SAMPLES = int(os.environ.get('N_SAMPLES', 1))


# -- Utility functions --

def convert_topk_prompt_logprobs(
    topk_prompt_logprobs_batch: List[List[Optional[List[tuple]]]],
    topk: int = 512,
) -> dict:
    """Convert vLLM topk_prompt_logprobs to CrossTokenLoss teacher_output format.

    Args:
        topk_prompt_logprobs_batch: List of per-input topk_prompt_logprobs.
            Each is List[Optional[List[(token_id, logprob)]]] of shape [seq_len, topk].
        topk: Number of top-k logits to extract.

    Returns:
        Dict with 'teacher_topk_logprobs' [batch, seq_len, topk] and
        'teacher_topk_indices' [batch, seq_len, topk] tensors.
    """
    batch_logprobs = []
    batch_indices = []
    for seq_topk in topk_prompt_logprobs_batch:
        seq_logprobs = []
        seq_indices = []
        print(f'seq_topk: {seq_topk},')
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


# -- Dataset --

def create_dataset():
    """Create a prompt-only dataset for on-policy distillation.

    The dataset only contains prompts; the student model generates completions
    on-the-fly. The teacher model computes logprobs on student-generated sequences.
    """
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(10000)))
    dataset.set_template('Template', model_id=STUDENT_MODEL_ID, max_length=MAX_LENGTH_SEQ)
    dataset.encode(load_from_cache_file=True)
    return dataset


# -- Training --

def train():
    """Main training loop for Cross-Token knowledge distillation."""
    import time
    start_time = time.perf_counter()
    print('Recording start time')

    # Initialize device groups for On-Policy mode
    device_groups = [
        DeviceGroup(name='student_model', ranks=MODEL_GPUS, device_type='npu'),
        DeviceGroup(name='student_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
        DeviceGroup(name='teacher_sampler', ranks=SAMPLER_GPUS, device_type='npu'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)

    twinkle.initialize(
        mode='ray',
        nproc_per_node=NUM_GPUS,
        groups=device_groups,
    )
    elapsed = time.perf_counter() - start_time
    print(f"initialize elapsed: {elapsed:.6f}s")
    start_time = time.perf_counter()

    # -- Student model (trainable) --
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
    student_model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    student_model.set_optimizer('AdamW', lr=LEARNING_RATE, weight_decay=0.01)
    student_model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=LEARNING_RATE * 0.1)

    # -- Configure CrossTokenLoss --
    from transformers import AutoTokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, trust_remote_code=True)
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
        vocab_topk=TOPK,
        device=torch.device('npu:0'),
    )
    student_model.set_loss(loss_fn, adapter_name=ADAPTER_NAME)
    student_model.set_template('Template', model_id=STUDENT_MODEL_ID, adapter_name=ADAPTER_NAME)
    elapsed = time.perf_counter() - start_time
    print(f"loss_fn init: {elapsed:.6f}s")
    start_time = time.perf_counter()

    # Log configuration
    logger.info(f'GPU Configuration: MODEL_GPUS={MODEL_GPUS}, SAMPLER_GPUS={SAMPLER_GPUS}')
    logger.info(f'Total GPUs required: {NUM_GPUS} (On-Policy: student_model + student_sampler + teacher_sampler)')

    # Log projection matrix statistics
    stats = loss_fn.get_mapping_statistics()
    logger.info(f'CrossToken Projection Matrix Statistics: {stats}')

    coverage_ratio = stats['exact_matched'] / stats['total_student_tokens']
    logger.info(f'Vocabulary coverage ratio: {coverage_ratio:.2%}')
    if coverage_ratio < 0.3:
        logger.warning(f"Low vocabulary coverage ({coverage_ratio:.2%}), consider using models with similar tokenizers")

    if LOSS_TYPE == 'hkl':
        logger.info(f'H-KL mode: gamma_kl={GAMMA_KL}, gamma_uld={GAMMA_ULD}')
        logger.info(f'  Unmatched tokens (ULD): {stats["unmatched"]}/{stats["total_student_tokens"]}')

    # -- Student vLLM sampler (for on-policy generation) --
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
    student_sampler.set_template('Template', model_id=STUDENT_MODEL_ID)

    # -- Teacher TransformersModel (for full logits computation) --
    teacher_model = TransformersModel(
        model_id=TEACHER_MODEL_ID,
        device_mesh=sampler_mesh,
        remote_group='teacher_sampler',
    )

    # 添加以下配置
    teacher_model.set_template('Template', model_id=TEACHER_MODEL_ID)

    # -- Checkpoint manager for weight sync --
    ckpt_manager = CheckpointEngineManager(model=student_model, sampler=student_sampler)

    # -- DataLoader --
    dataloader = DataLoader(
        dataset=create_dataset(),
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='student_model',
    )

    logger.info(get_device_placement())
    logger.info(f'CrossToken Training | student={STUDENT_MODEL_ID}  teacher={TEACHER_MODEL_ID}')
    logger.info(f'  loss_type={LOSS_TYPE}  T={TEMPERATURE}  topk={TOPK}')
    logger.info(f'  beta={BETA}  gamma={GAMMA}  max_length={MAX_LENGTH}')
    if LOSS_TYPE == 'hkl':
        logger.info(f'  gamma_kl={GAMMA_KL}  gamma_uld={GAMMA_ULD}')
    logger.info(f'  batch_size={BATCH_SIZE}  lr={LEARNING_RATE}  max_steps={MAX_STEPS}')

    # -- Training Loop (On-Policy) --
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        if callable(batch):
            batch = batch()

        # Step 1: Sync student model weights to student sampler
        ckpt_manager.sync_weights(merge_and_sync=False)
        student_sampler.reset_prefix_cache()
        student_sampler.reset_encoder_cache()

        # Step 2: Student vLLM generates completions
        sample_response = student_sampler.sample(
            batch,
            SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=1.0, num_samples=N_SAMPLES),
        )

        # --- Print student responses as text ---
        print("\n" + "=" * 80)
        print(f"[Step {optim_step}] STUDENT GENERATED RESPONSES:")
        print("=" * 80)
        for i, resp in enumerate(sample_response):
            for j, seq in enumerate(resp.sequences):
                # seq.tokens 只包含新生成的 token（不含 prompt）
                gen_text = student_tokenizer.decode(seq.tokens, skip_special_tokens=False)
                # new_input_feature 中的 input_ids 包含 prompt + 新生成 token
                full_text = student_tokenizer.decode(seq.new_input_feature['input_ids'], skip_special_tokens=False)
                print(f"\n--- Student sample[{i}].sequence[{j}] ---")
                # print(f"  FULL TEXT (prompt + generation): {full_text}")
                # print(f"  GENERATED ONLY (from seq.tokens): {gen_text}")
                # if seq.decoded is not None:
                #     print(f"  seq.decoded: {seq.decoded}")
                # print(f"  stop_reason: {seq.stop_reason}")
                # 打印生成 token 的概率序列（每个 token 位置的 top-k logprobs）
                if seq.logprobs is not None:
                    print(f"  GENERATED LOGPROBS (per token position, top-5 shown):")
                    for pos, pos_logprobs in enumerate(seq.logprobs):
                        top5 = sorted(pos_logprobs, key=lambda x: x[1], reverse=True)[:5]
                        tokens_str = ", ".join([f"(id={tid}, logprob={lp:.4f})" for tid, lp in top5])
                        print(f"    pos[{pos}]: {tokens_str}")
                else:
                    print(f"  GENERATED LOGPROBS: None")

        # Extract the generated sequences (prompt + student-generated response)
        input_data = [seq.new_input_feature for response in sample_response for seq in response.sequences]

        # Re-encode student-generated text with teacher tokenizer.
        # For same-tokenizer-family scenarios, decode-encode is a lossy round-trip
        # that can produce token sequences of different lengths, causing the loss
        # to compare misaligned positions (inflated KL). Instead, reuse the
        # student token ids directly when the tokenizers share a common vocab.
        # The CrossTokenLoss projection matrix handles the cross-tokenizer mapping.
        teacher_input_data = []
        for feat in input_data:
            student_ids = feat['input_ids']
            if isinstance(student_ids, (list, tuple)):
                student_ids = torch.tensor(student_ids, dtype=torch.long)
            teacher_input_data.append({
                'input_ids': student_ids,
                'labels': student_ids.clone().detach() if hasattr(student_ids, 'clone') else student_ids,
            })

        # Step 3: Teacher computes full logits on student-generated sequences
        # Use TransformersModel to get full logits (no vLLM 64 limit)
        # Note: forward_only is a remote_function, need to call it to get the actual result

        # 添加详细的输入数据调试信息
        print(f'\n{"="*80}')
        print(f'[Step {optim_step}] TEACHER INPUT DATA DEBUG:')
        print(f'{"="*80}')
        print(f'teacher_input_data length: {len(teacher_input_data)}')
        for i, feat in enumerate(teacher_input_data[:2]):  # 只打印前两个样本的详细信息
            print(f'  Sample {i}:')
            print(f'    input_ids shape: {feat["input_ids"].shape if isinstance(feat["input_ids"], torch.Tensor) else len(feat["input_ids"])}')
            print(f'    labels shape: {feat["labels"].shape if isinstance(feat["labels"], torch.Tensor) else len(feat["labels"])}')
            print(f'    input_ids[:10]: {feat["input_ids"][:10] if hasattr(feat["input_ids"], "__getitem__") else "N/A"}')
            print(f'    labels[:10]: {feat["labels"][:10] if hasattr(feat["labels"], "__getitem__") else "N/A"}')

        # 确保启用 return_logits=True 以获取 logits
        teacher_outputs = teacher_model.forward_only(
            inputs=teacher_input_data,
            return_logits=True,  # 必须启用 return_logits
            temperature=1.0,
            disable_lora=True,
            adapter_name=''
        )
        teacher_outputs = teacher_outputs()  # 调用函数获取实际结果

        # Step 4: Prepare teacher output with full logits
        # Use input_ids from teacher_input_data (actual tokens used by TransformersModel).
        teacher_prompt_ids = [torch.tensor(feat['input_ids'], dtype=torch.long)
                              for feat in teacher_input_data]
        teacher_padded = rnn_utils.pad_sequence(
            teacher_prompt_ids,
            batch_first=True,
            padding_value=0,
        )
        # Use raw logits instead of log probabilities for NeMo-style P-KL
        teacher_logits = teacher_outputs['logits']  # Get raw logits from teacher model
        teacher_output = {
            'teacher_logits_group': [teacher_logits],  # Use raw logits for P-KL
            'teacher_input_ids_group': [teacher_padded],
        }

        # Step 5: Student forward + CrossToken backward
        student_model.forward_backward(
            inputs=input_data,
            adapter_name=ADAPTER_NAME,
            return_logits=True,
            **teacher_output,
        )

        student_model.clip_grad_and_step(adapter_name=ADAPTER_NAME)

        # Logging
        if optim_step > 0 and optim_step % 2 == 0:
            metric = student_model.calculate_metric(is_training=True, adapter_name=ADAPTER_NAME)
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {metric}')

        # Checkpoint
        if optim_step > 0 and optim_step % 100 == 0:
            student_model.save(f'cross-token-ckpt-{optim_step}', adapter_name=ADAPTER_NAME)

        optim_step += 1

    # Save final checkpoint
    student_model.save('cross-token-final', adapter_name=ADAPTER_NAME)
    logger.info('CrossToken training completed.')


if __name__ == '__main__':
    train()