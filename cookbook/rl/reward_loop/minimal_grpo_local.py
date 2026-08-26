"""Minimal local-mode GRPO example using the reward loop (no twinkle-server).

Runs entirely with twinkle's local components (``twinkle.initialize`` +
local vLLM sampler + TransformersModel/MegatronModel), so it does **not**
require the twinkle-server deployment and therefore has **no tinker
dependency**.  The reward computation is delegated to
``AsyncRewardPipeline`` with a double-buffer timeline: submit batch k
immediately, collect and train batch k-1 while batch k is being rewarded.
"""
from __future__ import annotations

import os
from typing import Any, List

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, user_data_get
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward_loop import AsyncRewardPipeline, RewardItem
from twinkle.sampler import vLLMSampler
from twinkle.metric import CompletionRewardMetric
from twinkle.preprocessor.llm import GSM8KProcessor

logger = get_logger()

MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = os.environ.get('TWINKLE_USE_MEGATRON', '0') == '1'
# Qwen3.5/3.6 are multimodal (vision tower); other models use the plain
# chat template.  Override explicitly with TWINKLE_TEMPLATE_CLS when needed.
_IS_MULTIMODAL_QWEN = 'Qwen3.5' in MODEL_ID or 'Qwen3.6' in MODEL_ID
TEMPLATE_CLS = os.environ.get(
    'TWINKLE_TEMPLATE_CLS',
    'Qwen3_5Template' if _IS_MULTIMODAL_QWEN else 'Template',
)

MODEL_GPUS = int(os.environ.get('TWINKLE_MODEL_GPUS', '1'))
SAMPLER_GPUS = int(os.environ.get('TWINKLE_SAMPLER_GPUS', '1'))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('TWINKLE_NUM_GENERATIONS', '4'))
MAX_NEW_TOKENS = int(os.environ.get('TWINKLE_MAX_NEW_TOKENS', '1024'))
LEARNING_RATE = float(os.environ.get('TWINKLE_LEARNING_RATE', '1e-5'))
MAX_STEPS = int(os.environ.get('TWINKLE_MAX_STEPS', '20'))
BATCH_SIZE = int(os.environ.get('TWINKLE_BATCH_SIZE', '4'))
MINI_BATCH_SIZE = int(os.environ.get('TWINKLE_MINI_BATCH_SIZE', '4'))
MICRO_BATCH_SIZE = int(os.environ.get('TWINKLE_MICRO_BATCH_SIZE', '2'))
ADAPTER_NAME = os.environ.get('TWINKLE_ADAPTER_NAME', 'local-grpo')
SAVE_STEPS = int(os.environ.get('TWINKLE_SAVE_STEPS', '10'))
REWARD_MODE = os.environ.get('TWINKLE_REWARD_MODE', 'async')
REWARD_NUM_WORKERS = int(os.environ.get('TWINKLE_REWARD_NUM_WORKERS', '2'))
REWARD_BACKLOG = int(os.environ.get('TWINKLE_REWARD_BACKLOG', '2'))


def create_gsm8k_dataset() -> Dataset:
    """Build the GSM8K dataset WITHOUT ``encode()``.

    Rows stay plain dicts carrying ``messages`` + ``user_data`` (ground truth)
    so reward adapters can resolve answers and ground truth directly.  The
    sampler encodes them on the fly (its template is set below); the returned
    ``new_input_feature`` keeps ``messages``/``user_data`` and is directly
    consumable by ``model.forward_backward``.
    """
    dataset = Dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template(TEMPLATE_CLS, model_id=MODEL_ID, max_length=400)
    return dataset


def gsm8k_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict):
    """Adapt the batch GSM8K reward to reward_loop's scalar contract.

    ``extra_info['prompt']`` is the un-encoded dataset row (``messages`` +
    ``user_data``), so the trajectory can be reconstructed here directly.
    """
    prompt = extra_info.get('prompt') if isinstance(extra_info, dict) else {}
    prompt = dict(prompt) if isinstance(prompt, dict) else {}
    messages = list(prompt.get('messages') or [])
    messages.append({'role': 'assistant', 'content': solution_str})
    trajectory = {**prompt, 'messages': messages}
    user_data = list(trajectory.get('user_data') or [])
    if user_data_get(user_data, 'ground_truth', None) in (None, ''):
        user_data.append(('ground_truth', str(ground_truth)))
    trajectory['user_data'] = user_data
    return GSM8KAccuracyReward()([trajectory])[0], {'data_source': data_source}


def make_reward_items(
    prompts: List[dict[str, Any]],
    sample_responses: List[Any],
    step: int,
    num_generations: int,
) -> List[RewardItem]:
    """Build one RewardItem per generated sequence, in prompt-major order.

    ``sample_responses`` is the expanded (prompt-major) sampler output, one
    response per prompt copy with a single sequence each.
    """
    items: List[RewardItem] = []
    for sequence_index, response in enumerate(sample_responses):
        prompt_index = sequence_index // num_generations
        prompt = prompts[prompt_index]
        for sequence in response.sequences:
            items.append(RewardItem(
                item_id=f'step-{step}/sample-{sequence_index}',
                data_source='gsm8k',
                solution_str=sequence.decoded or '',
                ground_truth=str(user_data_get(prompt.get('user_data'), 'ground_truth', '')),
                extra_info={'prompt': prompt},
            ))
    return items


def main() -> None:
    if min(MAX_STEPS, BATCH_SIZE, NUM_GENERATIONS, REWARD_NUM_WORKERS, REWARD_BACKLOG) <= 0:
        raise ValueError('MAX_STEPS, BATCH_SIZE, NUM_GENERATIONS, REWARD_NUM_WORKERS '
                         'and REWARD_BACKLOG must be positive')
    if REWARD_MODE not in ('async', 'sync'):
        raise ValueError("TWINKLE_REWARD_MODE must be 'async' or 'sync'")

    # Separate device groups so the model and the vLLM sampler use distinct GPUs.
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    # mode='ray' starts a local Ray cluster on this machine automatically —
    # no twinkle-server, no tinker.
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj',
        ],
        r=32, lora_alpha=64, lora_dropout=0.05,
    )
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', mixed_precision='bf16')
    else:
        # model_cls is intentionally omitted: TransformersModel resolves it
        # from the checkpoint's ``config.architectures`` (falling back to
        # AutoModelForCausalLM), so no hard-coded class name is needed.
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=1)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template(TEMPLATE_CLS, model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 4496,
            'max_lora_rank': 32,
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_CLS, model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1)

    pipeline = AsyncRewardPipeline(
        num_workers=REWARD_NUM_WORKERS,
        mode=REWARD_MODE,
        backlog=REWARD_BACKLOG,
        worker_kwargs={'compute_score': gsm8k_score},
    )

    optim_step = 0
    logger.info(get_device_placement())
    try:
        last_handle = None
        last_payload = None  # (input_data, old_logps, completion_lengths, step)
        for step, batch in enumerate(dataloader):
            if optim_step >= MAX_STEPS:
                break
            metrics.reset()
            global_prompts = batch if isinstance(batch, list) else [batch]
            ckpt_manager.sync_weights(merge_and_sync=False)
            sampler.reset_prefix_cache()

            expand_prompts = []
            for prompt in global_prompts:
                expand_prompts.extend([prompt] * NUM_GENERATIONS)

            # Batch sampling: prompt-major order (prompt x NUM_GENERATIONS).
            sample_responses = sampler.sample(expand_prompts, sampling_params)
            items = make_reward_items(global_prompts, sample_responses, step, NUM_GENERATIONS)

            # Fire-and-forget: submit the current batch immediately.
            handle = pipeline.submit(items)

            if last_handle is not None:
                # The previous batch finished in the background; collect is
                # nearly free here, then train on it.
                reward_results = pipeline.collect(last_handle)
                _train_batch(
                    model=model,
                    advantage_fn=advantage_fn,
                    metrics=metrics,
                    input_data=last_payload[0],
                    old_logps=last_payload[1],
                    completion_lengths=last_payload[2],
                    rewards=[result.reward_score for result in reward_results],
                    step=last_payload[3],
                )
                optim_step += 1

            last_handle = handle
            last_payload = (
                _collect_input_data(sample_responses),
                _collect_old_logps(sample_responses),
                _collect_completion_lengths(sample_responses),
                step,
            )

        # Flush the final in-flight batch.
        if last_handle is not None and optim_step < MAX_STEPS:
            reward_results = pipeline.collect(last_handle)
            _train_batch(
                model=model,
                advantage_fn=advantage_fn,
                metrics=metrics,
                input_data=last_payload[0],
                old_logps=last_payload[1],
                completion_lengths=last_payload[2],
                rewards=[result.reward_score for result in reward_results],
                step=last_payload[3],
            )
            optim_step += 1
    finally:
        pipeline.close()

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save(f'grpo-gsm8k-checkpoint')


def _collect_input_data(sample_responses: List[Any]) -> List[dict[str, Any]]:
    return [
        sequence.new_input_feature
        for response in sample_responses
        for sequence in response.sequences
    ]


def _collect_old_logps(sample_responses: List[Any]) -> List[List[float]]:
    return [
        [logprob[0][1] for logprob in sequence.logprobs]
        for response in sample_responses
        for sequence in response.sequences
    ]


def _collect_completion_lengths(sample_responses: List[Any]) -> List[int]:
    return [
        len(sequence.tokens)
        for response in sample_responses
        for sequence in response.sequences
    ]


def _train_batch(
    *,
    model,
    advantage_fn,
    metrics,
    input_data: List[dict[str, Any]],
    old_logps: List[List[float]],
    completion_lengths: List[int],
    rewards: List[float],
    step: int,
) -> None:
    """Compute group-normalized advantages and run one optimizer step per mini-batch."""
    advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()
    metrics.accumulate(
        completion_lengths=completion_lengths,
        rewards={'total': rewards},
    )

    total_completions = len(input_data)
    for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
        mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
        model.forward_backward(
            inputs=input_data[mb_start:mb_end],
            old_logps=old_logps[mb_start:mb_end],
            advantages=advantages[mb_start:mb_end],
            micro_batch_size=MICRO_BATCH_SIZE,
        )
        model.clip_grad_and_step()

    log_dict = metrics.calculate()
    log_dict.update(model.calculate_metric(is_training=True))
    logger.info(f'[Step {step}/{MAX_STEPS}] reward_mean={sum(rewards) / len(rewards):.4f} {log_dict}')


if __name__ == '__main__':
    main()