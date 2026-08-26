from __future__ import annotations

import asyncio
from concurrent.futures import Future

import pytest

from twinkle.data_format import SamplingParams
from twinkle.server.sampler.twinkle_handlers import _await_generation
from twinkle_agentic.async_rl.vllm_sampler_tq import (
    VLLMSamplerTQ,
    _GenerationSubmission,
    _dispatch_generation,
)


def _bare_sampler() -> VLLMSamplerTQ:
    sampler = object.__new__(VLLMSamplerTQ)
    sampler._generation_submissions = {}
    return sampler


def test_generation_dispatch_allows_one_prompt_with_multiple_dp_workers() -> None:
    assert VLLMSamplerTQ.submit_generation._dispatch is _dispatch_generation
    assert VLLMSamplerTQ.sample._dispatch == 'slice_dp'
    shards = [
        _dispatch_generation(
            3,
            worker_index,
            ('submission', [{'input_ids': [1]}], 'params'),
            {},
        )[0][1]
        for worker_index in range(3)
    ]

    assert shards == [[{'input_ids': [1]}], [], []]


def test_generation_dispatch_injects_prompt_index_base() -> None:
    inputs = [{'input_ids': [index]} for index in range(5)]
    bases = [
        _dispatch_generation(3, worker_index, ('submission', inputs, 'params'), {})[1]
        .get('_prompt_index_base')
        for worker_index in range(3)
    ]

    assert bases == [0, 1, 3]


def test_generation_submission_returns_before_generation_finishes() -> None:
    sampler = _bare_sampler()
    pending = Future()
    submitted_coroutines = []

    def submit(coro):
        submitted_coroutines.append(coro)
        coro.close()
        return pending

    sampler._submit_in_loop = submit

    result = sampler.submit_generation(
        'submission-1',
        [{'input_ids': [1]}],
        SamplingParams(max_tokens=4),
    )

    assert result == {'submission_id': 'submission-1', 'status': 'running'}
    assert not pending.done()
    assert len(submitted_coroutines) == 1
    assert sampler.get_generation_status('submission-1')['status'] == 'running'

    responses = [object()]
    pending.set_result(responses)
    assert sampler.get_generation_status('submission-1')['status'] == 'completed'
    assert sampler.collect_generation('submission-1') == responses
    assert 'submission-1' not in sampler._generation_submissions


def test_generation_keeps_one_response_per_prompt() -> None:
    sampler = _bare_sampler()
    sampler.template = None

    class FakeResponse:
        def __init__(self, value):
            self.sequences = [value]
            self.prompt_token_ids = None
            self.prompt_logprobs = None
            self.topk_prompt_logprobs = None

    async def sample_single(feat, _params, **_kwargs):
        await asyncio.sleep(0)
        return FakeResponse(feat['input_ids'][0])

    sampler._sample_single = sample_single
    responses = asyncio.run(
        sampler._generate_inputs(
            [{'input_ids': [10]}, {'input_ids': [20]}],
            SamplingParams(max_tokens=4, num_samples=2),
            adapter_name='',
            adapter_path=None,
            use_base_model=False,
        ))

    assert len(responses) == 2
    assert [len(response.sequences) for response in responses] == [2, 2]
    assert [response.sequences[0] for response in responses] == [10, 20]
    assert [response.sequences[1] for response in responses] == [10, 20]


def test_generation_failure_is_isolated_and_consumed() -> None:
    sampler = _bare_sampler()
    failed = Future()
    failed.set_exception(ValueError('bad prompt'))
    sampler._generation_submissions['failed'] = _GenerationSubmission(failed)

    state = sampler.get_generation_status('failed')
    assert state['status'] == 'failed'
    assert state['error'] == 'ValueError: bad prompt'

    with pytest.raises(ValueError, match='bad prompt'):
        sampler.collect_generation('failed')
    assert 'failed' not in sampler._generation_submissions


def test_generation_can_be_cancelled_without_waiting() -> None:
    sampler = _bare_sampler()
    pending = Future()
    sampler._generation_submissions['pending'] = _GenerationSubmission(pending)

    state = sampler.cancel_generation('pending')

    assert state == {'submission_id': 'pending', 'status': 'cancelled'}
    assert pending.cancelled()
    assert 'pending' not in sampler._generation_submissions


def test_all_generations_are_cancelled_on_shutdown() -> None:
    sampler = _bare_sampler()
    first = Future()
    second = Future()
    sampler._generation_submissions.update(
        first=_GenerationSubmission(first),
        second=_GenerationSubmission(second),
    )

    state = sampler.cancel_all_generations()

    assert state == {'submissions': 2, 'cancelled': 2}
    assert first.cancelled()
    assert second.cancelled()
    assert sampler._generation_submissions == {}


def test_generation_submission_tracks_per_sequence_completion() -> None:
    submission = _GenerationSubmission(Future())

    async def produce():
        tasks = [
            asyncio.create_task(asyncio.sleep(0, result=index))
            for index in range(3)
        ]
        submission.register(tasks, index_base=2)
        await asyncio.gather(*tasks)

    asyncio.run(produce())

    assert submission.total == 3
    assert submission.completed_indices() == [2, 3, 4]
    taken = submission.take([3, 2])
    assert [index for index, _response in taken] == [3, 2]
    assert [response for _index, response in taken] == [1, 0]
    assert submission.completed_indices() == [4]
    # Already-popped and unknown indices are skipped, not errored.
    assert submission.take([3, 99]) == []


def test_generation_status_reports_completed_samples() -> None:
    sampler = _bare_sampler()
    submission = _GenerationSubmission(Future())
    sampler._generation_submissions['batch'] = submission

    async def produce():
        tasks = [
            asyncio.create_task(asyncio.sleep(0, result=index))
            for index in range(4)
        ]
        submission.register(tasks, index_base=0)
        await asyncio.gather(*tasks)

    asyncio.run(produce())
    state = sampler.get_generation_status('batch')
    assert state['status'] == 'running'  # whole-group future still pending
    assert state['completed_samples'] == 4
    assert state['completed_indices'] == [0, 1, 2, 3]
    assert state['total_samples'] == 4


def test_collect_ready_samples_pops_completed_indices() -> None:
    sampler = _bare_sampler()
    submission = _GenerationSubmission(Future())
    sampler._generation_submissions['batch'] = submission

    async def produce():
        tasks = [
            asyncio.create_task(asyncio.sleep(0, result={'response': index}))
            for index in range(3)
        ]
        submission.register(tasks, index_base=0)
        await asyncio.gather(*tasks)

    asyncio.run(produce())
    collected = sampler.collect_ready_samples('batch', [1, 0])
    assert collected == [(1, {'response': 1}), (0, {'response': 0})]
    # Second collection of the same indices must be empty (pop semantics).
    assert sampler.collect_ready_samples('batch', [1, 0]) == []
    assert sampler.collect_ready_samples('batch', [2]) == [(2, {'response': 2})]


def test_native_prompt_group_sampling_requires_context_manager() -> None:
    sampler = _bare_sampler()
    sampler.context_manager = None

    with pytest.raises(RuntimeError, match='context_manager is required'):
        sampler.sample([], SamplingParams(max_tokens=4))


def test_server_waiter_admits_later_submission_before_first_finishes() -> None:

    class Sampler:

        def __init__(self):
            self.futures: dict[str, Future] = {}
            self.submission_order = []

        def submit_generation(self, submission_id, *_args, **_kwargs):
            self.submission_order.append(submission_id)
            self.futures[submission_id] = Future()

        def get_generation_status(self, submission_id):
            future = self.futures[submission_id]
            return {'status': 'completed' if future.done() else 'running'}

        def collect_generation(self, submission_id):
            return self.futures[submission_id].result()

        def cancel_generation(self, submission_id):
            self.futures.pop(submission_id, None)

    sampler = Sampler()

    async def run():
        sampler.submit_generation('first')
        sampler.submit_generation('second')
        first = asyncio.create_task(
            _await_generation(sampler, 'first'))
        second = asyncio.create_task(
            _await_generation(sampler, 'second'))
        while len(sampler.submission_order) < 2:
            await asyncio.sleep(0)
        assert not first.done()
        sampler.futures['second'].set_result(['short'])
        assert await second == ['short']
        assert not first.done()
        sampler.futures['first'].set_result(['long'])
        assert await first == ['long']

    asyncio.run(run())
    assert set(sampler.submission_order) == {'first', 'second'}


def test_server_waiter_retries_cancelled_status_poll() -> None:
    from ray.exceptions import TaskCancelledError

    class Sampler:

        def __init__(self):
            self.status_calls = 0
            self.cancelled = False

        def submit_generation(self, *_args, **_kwargs):
            return None

        def get_generation_status(self, _submission_id):
            self.status_calls += 1
            if self.status_calls == 1:
                raise TaskCancelledError()
            return {'status': 'completed'}

        def collect_generation(self, _submission_id):
            return ['completed']

        def cancel_generation(self, _submission_id):
            self.cancelled = True

    sampler = Sampler()
    sampler.submit_generation('submission')
    result = asyncio.run(
        _await_generation(sampler, 'submission'))

    assert result == ['completed']
    assert sampler.status_calls == 2
    assert not sampler.cancelled
