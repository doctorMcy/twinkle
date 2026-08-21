import asyncio
import importlib
from typing import Optional

from .data import RewardItem
from .default_score import compute_score as default_compute_score
from .reward_manager import get_reward_manager_cls

try:
    from twinkle.infra import remote_class
except ImportError:  # pragma: no cover
    remote_class = lambda **kwargs: (lambda cls: cls)


@remote_class(execute="all")
class RewardLoopWorker:
    def __init__(self, manager_name="naive", compute_score=None, custom_reward_function_path=None,
                 custom_reward_function_name="compute_score", unknown_rewards="warn", reward_kwargs=None,
                 max_rpm=None, max_tpm=None, max_concurrent=1, timeout=300.0, **kwargs):
        if compute_score is None and custom_reward_function_path:
            module = importlib.import_module(custom_reward_function_path)
            compute_score = getattr(module, custom_reward_function_name)
        self.unknown_rewards = unknown_rewards
        self.compute_score = compute_score or default_compute_score
        manager_cls = get_reward_manager_cls(manager_name)
        options = dict(reward_kwargs or {})
        if manager_name == "rate_limited":
            options.update(max_rpm=max_rpm, max_tpm=max_tpm, max_concurrent=max_concurrent, timeout=timeout)
        self.manager = manager_cls(compute_score=self.compute_score, **options)

    def compute_score_batch(self, items):
        return asyncio.run(self.manager.run_batch(items))
