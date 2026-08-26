import asyncio
import json
import queue
import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from twinkle_client.http import http_post
from twinkle_client.http.http_utils import _build_headers
from twinkle_client.types.sampler import AddAdapterResponse, SampleResponseModel, SetTemplateResponse
from peft import PeftConfig
from twinkle.data_format import Trajectory, InputFeature
from twinkle_client.common.json_utils import json_safe
from twinkle_client.types.component import DataRef


@dataclass
class StreamSample:
    """One finished generated sequence received from the streaming endpoint."""

    index: int
    row: Dict[str, Any]
    ref: DataRef
    done: int
    total: int


@dataclass
class StreamComplete:
    """Terminal event carrying the fully-filled DataRef."""

    ref: DataRef


# Intentionally does NOT subclass ``twinkle.sampler.base.Sampler``: importing
# that base pulls ``twinkle.sampler.__init__`` → ``VLLMEngine`` → torch + zmq,
# which the mock / CPU-only client environments don't have.
def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy arrays / torch tensors to JSON-serialisable lists.

    ``sample()`` accepts pre-encoded ``InputFeature`` dicts (e.g. from a multi-turn
    rollout's ``template.encode``) whose values are numpy arrays or torch tensors;
    these are not JSON-serialisable and would break the HTTP POST. Detection is by
    duck-typing (``.tolist()``) so this stays free of a hard torch/numpy import,
    honouring the CPU-only client contract noted above.
    """
    return json_safe(obj)


class vLLMSampler:
    """Client wrapper for Sampler that calls server HTTP endpoints.

    This client manages sampling operations and adapter synchronization with the sampler server.
    The server-side session (managed by TwinkleClient) keeps the sampler alive.
    """

    def __init__(self, model_id: str, **kwargs):
        """Create the sampler instance on server."""
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()
        from twinkle_client.data_plane import DataPlaneClient
        self.data_plane = DataPlaneClient(kwargs.pop('data_plane_url', None))

        self.adapter_name = None
        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.model_id = model_id
        self.server_url = f'{self.server_url}/sampler/{model_id}/twinkle'
        response = http_post(
            url=f'{self.server_url}/create',
            json_data=kwargs
        )
        response.raise_for_status()

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig, **kwargs) -> AddAdapterResponse:
        """Add a new adapter to the sampler."""
        if isinstance(config, PeftConfig):
            config = config.__dict__
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            json_data={'adapter_name': adapter_name, 'config': config, **kwargs}
        )
        response.raise_for_status()
        self.adapter_name = adapter_name
        return AddAdapterResponse(**response.json())

    def sample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Dict[str, Any]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Sample from the model.

        Args:
            inputs: List of Trajectory or InputFeature to sample from.
            sampling_params: Sampling parameters dict.
            adapter_name: Adapter name for LoRA inference.
            adapter_uri: Adapter URI (twinkle:// path or local path) for LoRA inference.
            num_samples: Number of completions to generate per prompt.

        Returns:
            SampleResponseModel with 'sequences' list, each containing tokens, logprobs, stop_reason.
        """
        sampling_params = dict(sampling_params or {})
        sampling_params['num_samples'] = num_samples
        json_data = {
            'inputs': _json_safe(inputs),
            'sampling_params': sampling_params,
            'adapter_name': adapter_name,
            'num_samples': num_samples,
        }
        if adapter_uri is not None:
            json_data['adapter_uri'] = adapter_uri

        response = http_post(
            url=f'{self.server_url}/sample',
            json_data=json_data
        )
        response.raise_for_status()
        return [SampleResponseModel(**r) for r in response.json()['samples']]

    def sample_to_data_plane(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> DataRef:
        """Generate complete prompt groups and keep their rows in the server DataPlane."""
        body = {
            'sampling_params': sampling_params,
            'adapter_name': adapter_name,
            'adapter_uri': adapter_uri,
            'policy_version': policy_version,
            'group_ids': group_ids,
            'num_samples': num_samples,
        }
        body['input_ref' if isinstance(inputs, DataRef) else 'inputs'] = (
            inputs.model_dump() if isinstance(inputs, DataRef) else _json_safe(inputs))
        response = http_post(
            url=f'{self.server_url}/sample_to_data_plane',
            json_data=json_safe(body),
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def asample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Dict[str, Any]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Asynchronous convenience wrapper for the materialized sample API."""
        return await asyncio.to_thread(
            self.sample,
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            num_samples=num_samples,
        )

    async def asample_to_data_plane(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> DataRef:
        """Asynchronously sample and return the opaque server-side result reference."""
        return await asyncio.to_thread(
            self.sample_to_data_plane,
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            policy_version=policy_version,
            group_ids=group_ids,
            num_samples=num_samples,
        )

    async def stream_sample_to_data_plane(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> AsyncIterator[Union[StreamSample, StreamComplete]]:
        """Stream each finished generated sequence as it completes.

        Yields one :class:`StreamSample` per sequence (row layout identical to
        :meth:`sample_to_data_plane`'s backed rows, with the pre-allocated
        DataRef in ``sample.ref``) and finally a :class:`StreamComplete`
        carrying the fully-filled DataRef.  HTTP and event errors raise.
        """
        body = {
            'sampling_params': sampling_params,
            'adapter_name': adapter_name,
            'adapter_uri': adapter_uri,
            'policy_version': policy_version,
            'group_ids': group_ids,
            'num_samples': num_samples,
        }
        body['input_ref' if isinstance(inputs, DataRef) else 'inputs'] = (
            inputs.model_dump() if isinstance(inputs, DataRef) else _json_safe(inputs))

        lines: queue.Queue = queue.Queue()
        sentinel = object()

        def _read() -> None:
            import requests
            try:
                response = requests.post(
                    f'{self.server_url}/sample_to_data_plane_stream',
                    json=json_safe(body),
                    headers=_build_headers(),
                    stream=True,
                    timeout=(60, 3600),
                )
                if response.status_code != 200:
                    detail = ''
                    try:
                        detail = response.json().get('detail', '') or response.text[:4000]
                    except Exception:
                        detail = response.text[:4000]
                    raise RuntimeError(
                        f'sample_to_data_plane_stream failed ({response.status_code}): {detail}')
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    lines.put(raw)
            except BaseException as error:  # noqa: BLE001
                lines.put(error)
            finally:
                lines.put(sentinel)

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        try:
            while True:
                item = await asyncio.to_thread(lines.get)
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise RuntimeError(f'streaming sample failed: {item}') from None
                event = json.loads(item)
                kind = event.get('event')
                if kind == 'error':
                    raise RuntimeError(event.get('error', 'unknown streaming error'))
                if kind == 'progress':
                    yield StreamSample(
                        index=event['index'],
                        row=event['row'],
                        ref=DataRef(**event['ref']),
                        done=event['done'],
                        total=event['total'],
                    )
                elif kind == 'ref':
                    yield StreamComplete(ref=DataRef(**event['ref']))
                else:
                    raise RuntimeError(f'unknown streaming event: {kind}')
        finally:
            thread.join(timeout=60)

    def unload_adapter_paths(self, adapter_paths: list[str]) -> None:
        """Evict policy snapshots that are no longer referenced by this client."""
        response = http_post(
            url=f'{self.server_url}/unload_adapter_paths',
            json_data={'adapter_paths': adapter_paths},
        )
        response.raise_for_status()

    def set_template(self, template_cls: str, adapter_name: str = '', **kwargs) -> SetTemplateResponse:
        """Set the template for encoding trajectories."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': adapter_name, **kwargs}
        )
        response.raise_for_status()
        return SetTemplateResponse(**response.json())
    
    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        response = http_post(
            url=f'{self.server_url}/apply_patch',
            json_data={'patch_cls': patch_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
