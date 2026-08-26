# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import pytest

from twinkle.server.data_plane.store import TQDataRefStore, _partition


@pytest.mark.asyncio
async def test_data_ref_round_trip_append_release_and_ref_isolation(monkeypatch) -> None:
    import transfer_queue as tq

    records = {}

    async def batch_put(*, keys, partition_id, fields, tags=None):
        storage_key = (partition_id, tuple(keys))
        if storage_key in records:
            records[storage_key].update(fields)
        else:
            records[storage_key] = fields.clone()
        if tags is not None:
            current = tag_records.setdefault(storage_key, [{} for _ in tags])
            for existing, update in zip(current, tags):
                existing.update(update)

    async def batch_get(*, keys, partition_id, select_fields):
        data = records[(partition_id, tuple(keys))]
        return data.select(*select_fields)

    async def clear(*, keys, partition_id):
        records.pop((partition_id, tuple(keys)))

    async def kv_list(partition_id):
        result = {}
        for (stored_partition, keys), tags in tag_records.items():
            if stored_partition == partition_id:
                result[stored_partition] = dict(zip(keys, tags))
        return result

    monkeypatch.setattr(tq, 'async_kv_batch_put', batch_put)
    monkeypatch.setattr(tq, 'async_kv_batch_get', batch_get)
    monkeypatch.setattr(tq, 'async_kv_clear', clear)
    monkeypatch.setattr(tq, 'async_kv_list', kv_list)

    # Bypass tq.init(): this test exercises only the DataRef mapping layer.
    store = TQDataRefStore.__new__(TQDataRefStore)
    tag_records = {}
    rows = [
        {'input_ids': [1, 2], 'answer': 'a'},
        {'input_ids': [3], 'answer': 'b'},
    ]
    tags = [{'group_id': 'g0', 'generation_idx': 0}, {'group_id': 'g0', 'generation_idx': 1}]
    ref = await store.put(rows, kind='train', tags=tags)

    assert ref.size == 2
    assert ref.fields == ['input_ids', 'answer']
    assert ref.num_tokens == 3
    assert await store.get(ref) == rows
    assert await store.get_tags(ref) == tags

    ref = await store.append(
        ref,
        [{'reward': 1.0}, {'reward': -1.0}],
        tags=[{'status': 'ready'}, {'status': 'ready'}],
    )
    assert ref.fields == ['input_ids', 'answer', 'reward']
    assert ref.num_tokens == 3
    assert await store.get(
        ref,
        fields=['answer', 'reward'],
    ) == [
        {'answer': 'a', 'reward': 1.0},
        {'answer': 'b', 'reward': -1.0},
    ]
    assert await store.get_tags(ref) == [
        {'group_id': 'g0', 'generation_idx': 0, 'status': 'ready'},
        {'group_id': 'g0', 'generation_idx': 1, 'status': 'ready'},
    ]

    ref = await store.append(
        ref,
        [{'input_ids': [4, 5, 6]}, {'input_ids': [7, 8]}],
    )
    assert ref.num_tokens == 5

    nested_ref = await store.put([
        {'train_input': {'input_ids': [1, 2, 3]}},
        {'train_input': {'input_ids': [4]}},
    ], kind='rollout')
    assert nested_ref.num_tokens == 4

    with pytest.raises(KeyError):
        await store.get(ref.model_copy(update={'ref_id': 'another-ref'}))

    await store.release(ref)
    await store.release(nested_ref)
    assert records == {}


@pytest.mark.asyncio
async def test_append_rejects_row_count_mismatch() -> None:
    from twinkle_client.types import DataRef

    store = TQDataRefStore.__new__(TQDataRefStore)
    ref = DataRef(ref_id='r', size=2, fields=['x'])
    with pytest.raises(ValueError, match='row count'):
        await store.append(ref, [{'reward': 1.0}])


def test_partition_is_stable_and_scoped_by_data_ref() -> None:
    from twinkle_client.types import DataRef

    first = DataRef(ref_id='a', size=1, fields=['x'])
    same = DataRef(ref_id='a', size=99, fields=['other'])
    other = DataRef(ref_id='b', size=1, fields=['x'])
    assert _partition(first) == _partition(same)
    assert _partition(first) != _partition(other)


@pytest.mark.asyncio
async def test_create_and_put_rows_fill_preallocated_ref_incrementally(monkeypatch) -> None:
    import transfer_queue as tq

    records = {}
    tag_records = {}

    async def batch_put(*, keys, partition_id, fields, tags=None):
        storage_key = (partition_id, tuple(keys))
        if storage_key in records:
            records[storage_key].update(fields)
        else:
            records[storage_key] = fields.clone()
        if tags is not None:
            current = tag_records.setdefault(storage_key, [{} for _ in tags])
            for existing, update in zip(current, tags):
                existing.update(update)

    async def kv_list(partition_id):
        result = {}
        for (stored_partition, keys), tags in tag_records.items():
            if stored_partition == partition_id:
                result[stored_partition] = dict(zip(keys, tags))
        return result

    monkeypatch.setattr(tq, 'async_kv_batch_put', batch_put)
    monkeypatch.setattr(tq, 'async_kv_list', kv_list)

    store = TQDataRefStore.__new__(TQDataRefStore)
    ref = await store.create(3, kind='rollout')
    assert ref.size == 3
    assert ref.fields == []
    assert ref.num_tokens == 0

    # Fill out of order: index 1 first, then 0, then 2 (with tags only on some).
    ref = await store.put_rows(
        ref,
        [{'input_ids': [3], 'decoded': 'b'}],
        [1],
        tags=[{'group_id': 'g0', 'generation_idx': 1}],
    )
    assert ref.fields == ['input_ids', 'decoded']
    assert ref.num_tokens == 1

    ref = await store.put_rows(
        ref,
        [{'input_ids': [1, 2], 'decoded': 'a'}],
        [0],
    )
    assert ref.fields == ['input_ids', 'decoded']
    assert ref.num_tokens == 3

    ref = await store.put_rows(
        ref,
        [{'input_ids': [4], 'decoded': 'c', 'reward': 0.0}],
        [2],
    )
    assert ref.fields == ['input_ids', 'decoded', 'reward']
    assert ref.num_tokens == 4

    keys = (f'twinkle-client/{ref.ref_id}', ('0', '1', '2'))
    data = records[keys]
    assert set(data.keys()) == {'input_ids', 'decoded', 'reward'}
    assert data['input_ids'].tolist() == [1, 2, 3, 4]
    assert list(data['decoded']) == ['a', 'b', 'c']
    assert data['reward'].tolist() == [0.0]
    assert await store.get_tags(ref) == [
        {'group_id': 'g0', 'generation_idx': 1},
        {},
        {},
    ]
    await store.release(ref)
    assert records == {}


@pytest.mark.asyncio
async def test_create_rejects_non_positive_size() -> None:
    store = TQDataRefStore.__new__(TQDataRefStore)
    with pytest.raises(ValueError, match='size'):
        await store.create(0)


@pytest.mark.asyncio
async def test_put_rows_validates_indices(monkeypatch) -> None:
    from twinkle_client.types import DataRef

    import transfer_queue as tq
    records = {}

    async def batch_put(*, keys, partition_id, fields, tags=None):
        records[(partition_id, tuple(keys))] = fields.clone()

    monkeypatch.setattr(tq, 'async_kv_batch_put', batch_put)

    store = TQDataRefStore.__new__(TQDataRefStore)
    ref = DataRef(ref_id='r', size=2, fields=[])

    with pytest.raises(ValueError, match='row count'):
        await store.put_rows(ref, [{'x': 1}], [0, 1])
    with pytest.raises(ValueError, match='unique'):
        await store.put_rows(ref, [{'x': 1}, {'x': 2}], [1, 1])
    with pytest.raises(ValueError, match='out of range'):
        await store.put_rows(ref, [{'x': 1}], [2])
    with pytest.raises(ValueError, match='out of range'):
        await store.put_rows(ref, [{'x': 1}], [-1])
    with pytest.raises(ValueError, match='must not be empty'):
        await store.put_rows(ref, [], [])
