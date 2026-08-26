# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json

import pytest

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.server.sampler.twinkle_handlers import (
    _poll_generation_events,
    _sample_model_to_row,
    _stream_sample_events,
)
from twinkle_client.types.component import DataRef

from twinkle_client.types.sampler import SampledSequenceModel, SampleResponseModel


def _model(index: int) -> SampleResponseModel:
    sequence = SampledSequenceModel(
        stop_reason='length',
        tokens=[index],
        logprobs=[[(index, -0.5)]],
        decoded=f'answer-{index}',
        new_input_feature={'input_ids': [1, 2]},
    )
    return SampleResponseModel(sequences=[sequence], prompt_logprobs=[], topk_prompt_logprobs=[])


class _FakeSampler:
    """Incremental completion: indices 0..1 round 1, 2..3 round 2, then done."""

    def __init__(self):
        self.status_calls = 0
        self.cancelled = False
        self.collected_indices: list[list[int]] = []
        self.responses = {index: _model(index) for index in range(4)}

    def get_generation_status(self, _submission_id):
        self.status_calls += 1
        if self.status_calls == 1:
            return {'status': 'running', 'completed_indices': [0, 1], 'total_samples': 4}
        return {'status': 'completed', 'completed_indices': [2, 3], 'total_samples': 4}

    def collect_ready_samples(self, _submission_id, indices):
        self.collected_indices.append(list(indices))
        return [(index, self.responses.pop(index)) for index in indices]

    def cancel_generation(self, _submission_id):
        self.cancelled = True


class _FakeDataPlane:
    def __init__(self, ref):
        self.ref = ref
        self.put_rows_calls: list[tuple[int, dict]] = []
        self.released = False

    async def put_rows(self, ref, rows, indices, *, tags=None):
        self.put_rows_calls.append((indices[0], ref))
        return ref

    async def release(self, ref):
        self.released = True


def test_sample_model_to_row_matches_batch_layout() -> None:
    row, tag = _sample_model_to_row(
        _model(0),
        group_id='g0',
        prompt_index=1,
        generation_idx=2,
        policy_version=3,
        adapter_uri='twinkle://p',
    )
    assert row['decoded'] == 'answer-0'
    assert row['train_input'] == {'input_ids': [1, 2]}
    assert tag['prompt_index'] == 1
    assert tag['generation_idx'] == 2
    assert tag['group_id'] == 'g0'
    assert tag['rollout_policy_version'] == 3


@pytest.mark.asyncio
async def test_stream_sample_events_emits_per_sample_progress_and_ref() -> None:
    sampler = _FakeSampler()
    ref = DataRef(ref_id='cohort', size=4, kind='rollout')
    data_plane = _FakeDataPlane(ref)

    async def collect():
        lines = []
        async for line in _stream_sample_events(
            sampler,
            data_plane,
            submission_id='sub',
            ref=ref,
            num_samples=2,
            total=4,
            group_ids=['g0', 'g1'],
            policy_version=0,
            adapter_uri=None,
        ):
            lines.append(json.loads(line))
        return lines

    events = asyncio.run(collect())

    progress = [event for event in events if event['event'] == 'progress']
    assert [event['index'] for event in progress] == [0, 1, 2, 3]
    assert [event['row']['decoded'] for event in progress] == [
        'answer-0', 'answer-1', 'answer-2', 'answer-3']
    assert [event['done'] for event in progress] == [1, 2, 3, 4]
    assert tuple(events[-1]) == ('event', 'ref')
    assert events[-1]['ref']['ref_id'] == 'cohort'
    assert events[-1]['ref']['size'] == 4
    # Indices were written in completion order, grouped by poll round.
    assert sampler.collected_indices == [[0, 1], [2, 3]]
    assert [(index, call_ref.ref_id) for index, call_ref in data_plane.put_rows_calls] == [
        (0, 'cohort'), (1, 'cohort'), (2, 'cohort'), (3, 'cohort')]
    assert not data_plane.released
    assert not sampler.cancelled


@pytest.mark.asyncio
async def test_stream_sample_events_releases_ref_on_failure() -> None:
    class FailingSampler:
        def get_generation_status(self, _submission_id):
            return {'status': 'failed', 'error': 'boom', 'completed_indices': []}

        def cancel_generation(self, _submission_id):
            self.cancelled = True

    sampler = FailingSampler()
    sampler.cancelled = False
    ref = DataRef(ref_id='cohort', size=4, kind='rollout')
    data_plane = _FakeDataPlane(ref)

    async def collect():
        lines = []
        async for line in _stream_sample_events(
            sampler,
            data_plane,
            submission_id='sub',
            ref=ref,
            num_samples=2,
            total=4,
            group_ids=['g0', 'g1'],
            policy_version=0,
            adapter_uri=None,
        ):
            lines.append(json.loads(line))
        return lines

    events = asyncio.run(collect())

    assert len(events) == 1
    assert events[0]['event'] == 'error'
    assert 'boom' in events[0]['error']
    assert sampler.cancelled
    assert data_plane.released


@pytest.mark.asyncio
async def test_poll_generation_events_yields_each_round_and_final_state() -> None:
    class Sampler:
        def __init__(self):
            self.status_calls = 0

        def get_generation_status(self, _submission_id):
            self.status_calls += 1
            return {
                'status': 'completed' if self.status_calls >= 2 else 'running',
                'completed_indices': [self.status_calls - 1],
            }

    sampler = Sampler()

    async def rounds():
        seen = []
        async for states in _poll_generation_events(sampler, 'sub'):
            seen.append(states)
        return seen

    rounds_seen = asyncio.run(rounds())
    assert [states[0]['status'] for states in rounds_seen] == ['running', 'completed']