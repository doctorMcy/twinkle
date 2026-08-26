from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from twinkle_client.sampler.vllm_sampler import StreamComplete, StreamSample
from twinkle_client.types.component import DataRef


MODULE_PATH = (
    Path(__file__).parents[2] / 'cookbook' / 'rl' / 'reward_loop' / 'minimal_grpo_true_stream.py'
)


def _load_module():
    spec = importlib.util.spec_from_file_location('minimal_grpo_true_stream', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSampler:

    def __init__(self, model_id):
        self.stream_calls = 0

    def set_template(self, *_args, **_kwargs):
        return None

    async def stream_sample_to_data_plane(self, prompts, **_kwargs):
        self.stream_calls += 1
        total = len(prompts) * 2
        for index in range(total):
            yield StreamSample(
                index=index,
                row={'decoded': f'answer-{index}'},
                ref=DataRef(ref_id=f'cohort-{self.stream_calls}', size=total),
                done=index + 1,
                total=total,
            )
        yield StreamComplete(ref=DataRef(
            ref_id=f'cohort-{self.stream_calls}', size=total, kind='rollout'))


class _FakeModel:

    def __init__(self, model_id):
        self.steps = 0
        self.forward_kwargs = []

    def add_adapter_to_model(self, *_args, **_kwargs):
        return None

    def set_loss(self, *_args, **_kwargs):
        return None

    def set_optimizer(self, *_args, **_kwargs):
        return None

    def set_processor(self, *_args, **_kwargs):
        return None

    def set_template(self, *_args, **_kwargs):
        return None

    async def forward_backward_from_data_plane(self, refs, **kwargs):
        self.forward_kwargs.append(kwargs)
        self.steps += 1

    async def clip_grad_and_step(self, **_kwargs):
        return None


class _FakeDataPlane:

    def __init__(self):
        self.released = []
        self.append_rows = []

    async def aappend(self, ref, rows, **_kwargs):
        self.append_rows.append(rows)
        return ref.model_copy(update={'fields': ['decoded', 'reward', 'advantage']})

    async def arelease(self, ref):
        self.released.append(ref)


class _FakeClient:

    def close(self):
        return None


def test_true_stream_submits_reward_per_sample_and_trains_once(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, 'MAX_STEPS', 1)
    monkeypatch.setattr(module, 'BATCH_SIZE', 2)
    monkeypatch.setattr(module, 'NUM_GENERATIONS', 2)
    monkeypatch.setattr(module, 'REWARD_NUM_WORKERS', 2)

    sampler = _FakeSampler('m')
    model = _FakeModel('m')
    data_plane = _FakeDataPlane()
    client = _FakeClient()
    monkeypatch.setattr(module, 'vLLMSampler', lambda _mid: sampler)
    monkeypatch.setattr(module, 'MultiLoraTransformersModel', lambda _mid: model)
    monkeypatch.setattr(module, 'DataPlaneClient', lambda: data_plane)
    monkeypatch.setattr(module, 'init_twinkle_client', lambda **_kw: client)

    asyncio.run(module.train())

    assert sampler.stream_calls == 1
    assert model.steps == 1
    assert model.forward_kwargs[0]['kwarg_fields'] == {
        'old_logps': 'sampled_logprobs',
        'advantages': 'advantage',
    }
    # One batch of reward rows appended to the streamed cohort ref.
    assert len(data_plane.append_rows) == 1
    assert len(data_plane.append_rows[0]) == 4
    assert all('reward' in row and 'advantage' in row for row in data_plane.append_rows[0])
    # Final and train refs released.
    assert len(data_plane.released) == 2
    assert all(ref.ref_id == 'cohort-1' for ref in data_plane.released)