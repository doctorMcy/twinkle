"""Minimal true-streaming (per-sample submission) GRPO example using the reward loop."""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from peft import LoraConfig

from twinkle.advantage import GRPOAdvantage
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.data_format import user_data_get
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward_loop import AsyncRewardPipeline, RewardItem
from twinkle_client import DataPlaneClient, init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler
from twinkle_client.sampler.vllm_sampler import StreamComplete, StreamSample

BASE_MODEL = os.environ.get("TWINKLE_MODEL_ID", "Qwen/Qwen3.5-4B")
MODEL_ID = f"ms://{BASE_MODEL}"
TEMPLATE_MODEL_ID = os.environ.get("TWINKLE_TEMPLATE_MODEL_ID", MODEL_ID)
TEMPLATE_CLS = os.environ.get(
    "TWINKLE_TEMPLATE_CLS",
    "Qwen3_5Template" if "Qwen3.5" in BASE_MODEL or "Qwen3.6" in BASE_MODEL else "Template",
)
ADAPTER_NAME = os.environ.get("TWINKLE_ADAPTER_NAME", "minimal-grpo-true-stream")
MAX_STEPS = int(os.environ.get("TWINKLE_MAX_STEPS", "2"))
BATCH_SIZE = int(os.environ.get("TWINKLE_BATCH_SIZE", "2"))
NUM_GENERATIONS = int(os.environ.get("TWINKLE_NUM_GENERATIONS", "2"))
REWARD_NUM_WORKERS = int(os.environ.get("TWINKLE_REWARD_NUM_WORKERS", "2"))


async def _call(method, *args, **kwargs):
    """Call either sync or async client methods without blocking the loop."""
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    value = await asyncio.to_thread(method, *args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def create_dataset() -> Dataset:
    dataset = Dataset(DatasetMeta("ms://modelscope/gsm8k", subset_name="main", split="train"))
    dataset.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID, max_length=2048, enable_thinking=False)
    dataset.map(GSM8KProcessor(system="Put the final answer within \\boxed{}."))
    dataset.encode(add_generation_prompt=True)
    return dataset


def gsm8k_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict):
    """Adapt Twinkle's trajectory reward to reward_loop's scalar contract."""
    prompt = extra_info.get("prompt") if isinstance(extra_info, dict) else {}
    prompt = dict(prompt) if isinstance(prompt, dict) else {}
    messages = list(extra_info.get("messages") or prompt.get("messages") or [])
    messages.append({"role": "assistant", "content": solution_str})
    trajectory = {**prompt, "messages": messages}
    user_data = list(trajectory.get("user_data") or [])
    if user_data_get(user_data, "ground_truth", None) in (None, ""):
        user_data.append(("ground_truth", str(ground_truth)))
    trajectory["user_data"] = user_data
    return GSM8KAccuracyReward()([trajectory])[0], {"data_source": data_source}


def _ground_truth(prompt: dict[str, Any]) -> str:
    return str(prompt.get("ground_truth") or user_data_get(prompt.get("user_data"), "ground_truth", ""))


async def _reward_item(
    event: StreamSample,
    step: int,
    prompts: list[dict[str, Any]],
    num_generations: int,
) -> RewardItem:
    prompt_index, generation_idx = divmod(event.index, num_generations)
    prompt = prompts[prompt_index]
    return RewardItem(
        item_id=f"step-{step}/sample-{event.index}",
        data_source="gsm8k",
        solution_str=str(event.row.get("decoded") or ""),
        ground_truth=_ground_truth(prompt),
        extra_info={"prompt": prompt, "messages": prompt.get("messages") or []},
    )


async def train() -> None:
    if min(MAX_STEPS, BATCH_SIZE, NUM_GENERATIONS) <= 0:
        raise ValueError("MAX_STEPS, BATCH_SIZE, and NUM_GENERATIONS must be positive")
    total_per_step = BATCH_SIZE * NUM_GENERATIONS

    client = init_twinkle_client(
        base_url=os.environ.get("TWINKLE_SERVER_URL", "http://localhost:8000"),
        api_key=os.environ.get("TWINKLE_SERVER_TOKEN", "EMPTY_TOKEN"),
    )
    # Every per-sample submission is one in-flight handle, so the backlog must
    # cover a whole step's samples (the true-streaming tradeoff).
    pipeline = AsyncRewardPipeline(
        num_workers=REWARD_NUM_WORKERS,
        mode="async",
        backlog=total_per_step,
        worker_kwargs={"compute_score": gsm8k_score},
    )
    refs = []
    try:
        model = MultiLoraTransformersModel(MODEL_ID)
        sampler = vLLMSampler(MODEL_ID)
        data_plane = DataPlaneClient()
        model.add_adapter_to_model(
            ADAPTER_NAME,
            LoraConfig(target_modules="all-linear", r=8, lora_alpha=32, lora_dropout=0.05),
        )
        model.set_loss("GRPOLoss", epsilon=0.2, beta=0.0)
        model.set_optimizer("AdamW", lr=2e-5)
        model.set_processor("InputProcessor", padding_free=True)
        model.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID)
        sampler.set_template(TEMPLATE_CLS, model_id=TEMPLATE_MODEL_ID)

        dataloader = DataLoader(dataset=create_dataset(), batch_size=BATCH_SIZE, num_workers=0)
        for step, batch in enumerate(dataloader):
            if step >= MAX_STEPS:
                break
            prompts = [dict(row) for row in batch]
            handles = []
            final_ref = None

            # Each StreamSample is one finished sequence: submit its reward
            # immediately, before the rest of the group has finished sampling.
            async for event in sampler.stream_sample_to_data_plane(
                prompts,
                adapter_name=ADAPTER_NAME,
                sampling_params={"max_tokens": 1024, "temperature": 1.0, "top_p": 0.95, "logprobs": 1},
                num_samples=NUM_GENERATIONS,
            ):
                if isinstance(event, StreamSample):
                    item = await _reward_item(event, step, prompts, NUM_GENERATIONS)
                    handle = await asyncio.to_thread(pipeline.submit, [item])
                    handles.append(handle)
                elif isinstance(event, StreamComplete):
                    final_ref = event.ref

            if final_ref is None or len(handles) != total_per_step:
                raise RuntimeError(
                    f"step {step} expected {total_per_step} streamed samples and a final ref, "
                    f"got {len(handles)} samples and final_ref={final_ref is not None}")

            # Collect per-sample rewards; index alignment is order-independent.
            results = []
            for handle in handles:
                results.extend(await asyncio.to_thread(pipeline.collect, handle))
            by_index: dict[int, float] = {}
            for result in results:
                index = int(result.item_id.rsplit("/sample-", 1)[1])
                if index in by_index:
                    raise RuntimeError(f"duplicate reward index {index}")
                by_index[index] = result.reward_score
            if set(by_index) != set(range(total_per_step)):
                raise RuntimeError(f"reward index mismatch: expected 0..{total_per_step - 1}, got {sorted(by_index)}")
            rewards = [by_index[index] for index in range(total_per_step)]

            advantages = await asyncio.to_thread(
                GRPOAdvantage(), rewards, num_generations=NUM_GENERATIONS)
            train_ref = await data_plane.aappend(
                final_ref,
                [{"reward": float(reward), "advantage": float(advantage)}
                 for reward, advantage in zip(rewards, advantages)],
            )
            refs.append(final_ref)
            try:
                await _call(
                    model.forward_backward_from_data_plane,
                    [train_ref],
                    input_field="train_input",
                    kwarg_fields={"old_logps": "sampled_logprobs", "advantages": "advantage"},
                    dynamic_batching=True,
                    micro_batch_size=4,
                    max_tokens_per_micro_batch=4096,
                )
                await _call(model.clip_grad_and_step, max_grad_norm=1.0)
            finally:
                await data_plane.arelease(train_ref)
                await data_plane.arelease(final_ref)
                refs.remove(final_ref)
            print(f"step={step} reward_mean={sum(rewards) / len(rewards):.4f}")
    finally:
        if "data_plane" in locals():
            for ref in refs:
                await data_plane.arelease(ref)
        pipeline.close()
        client.close()


if __name__ == "__main__":
    asyncio.run(train())