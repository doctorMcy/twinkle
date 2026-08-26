# Copyright (c) ModelScope Contributors. All rights reserved.
"""Internal HTTP adapter used by Model and Sampler component deployments."""
from __future__ import annotations

from typing import Any

import httpx

from twinkle_client.http.headers import build_routing_headers
from twinkle_client.types.component import DataRef


class DataPlaneProxy:

    def __init__(self, base_url: str | None):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.client = httpx.AsyncClient(timeout=None) if self.base_url else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    async def get(
        self,
        ref: DataRef,
        *,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required when a component request uses input_ref')
        response = await self.client.post(
            f'{self.base_url}/twinkle/get',
            json={'ref': ref.model_dump(), 'fields': fields},
            headers=build_routing_headers(f'data-ref-{ref.ref_id}'),
        )
        response.raise_for_status()
        return response.json()['rows']

    async def put(
        self,
        rows: list[dict[str, Any]],
        *,
        kind: str,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required to store component output')
        response = await self.client.post(
            f'{self.base_url}/twinkle/put',
            json={'rows': rows, 'kind': kind, 'tags': tags},
            headers=build_routing_headers(f'data-put-{kind}'),
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def append(
        self,
        ref: DataRef,
        rows: list[dict[str, Any]],
        *,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required to append component output')
        response = await self.client.post(
            f'{self.base_url}/twinkle/append',
            json={
                'ref': ref.model_dump(),
                'rows': rows,
                'tags': tags,
            },
            headers=build_routing_headers(f'data-append-{ref.ref_id}'),
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def create(
        self,
        size: int,
        *,
        kind: str = 'data',
    ) -> DataRef:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required to create component output refs')
        response = await self.client.post(
            f'{self.base_url}/twinkle/create',
            json={'size': size, 'kind': kind},
            headers=build_routing_headers(f'data-create-{kind}'),
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def put_rows(
        self,
        ref: DataRef,
        rows: list[dict[str, Any]],
        indices: list[int],
        *,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required to store component output')
        response = await self.client.post(
            f'{self.base_url}/twinkle/put_rows',
            json={
                'ref': ref.model_dump(),
                'rows': rows,
                'indices': indices,
                'tags': tags,
            },
            headers=build_routing_headers(f'data-put-rows-{ref.ref_id}'),
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def release(
        self,
        ref: DataRef,
    ) -> None:
        if self.client is None or self.base_url is None:
            raise RuntimeError('data_plane_url is required to release component output')
        response = await self.client.post(
            f'{self.base_url}/twinkle/release',
            json={'ref': ref.model_dump()},
            headers=build_routing_headers(f'data-release-{ref.ref_id}'),
        )
        response.raise_for_status()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
