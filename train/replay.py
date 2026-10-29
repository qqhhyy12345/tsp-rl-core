"""经验回放池（预分配环形缓冲）。"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """固定容量的环形缓冲区。

    用 numpy 预分配，避免 Python list 的内存碎片；
    采样后再一次性转成 torch 张量。
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device, n_cities: int | None = None):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.device = device
        # next_visited 的宽度是城市数 N，不是 state_dim 的一半 ——
        # 状态改成 3N 之后 state_dim // 2 就不再等于 N 了，必须显式传。
        self.n_cities = int(n_cities) if n_cities is not None else self.state_dim // 2

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_visited = np.zeros((self.capacity, self.n_cities), dtype=bool)
        self.dones = np.zeros(self.capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    def push(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_visited: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """批量写入（自动处理环形回绕）。"""
        k = states.shape[0]
        if k == 0:
            return
        if k > self.capacity:
            states, actions, rewards = states[-self.capacity :], actions[-self.capacity :], rewards[-self.capacity :]
            next_states, next_visited, dones = (
                next_states[-self.capacity :],
                next_visited[-self.capacity :],
                dones[-self.capacity :],
            )
            k = self.capacity

        idx = (self.ptr + np.arange(k)) % self.capacity
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states
        self.next_visited[idx] = next_visited
        self.dones[idx] = dones

        self.ptr = (self.ptr + k) % self.capacity
        self.size = min(self.size + k, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        """随机采样一批，返回 torch 张量元组。"""
        idx = rng.integers(0, self.size, size=batch_size)
        to = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=self.device)  # noqa: E731
        return (
            to(self.states[idx], torch.float32),
            to(self.actions[idx], torch.long),
            to(self.rewards[idx], torch.float32),
            to(self.next_states[idx], torch.float32),
            to(self.next_visited[idx], torch.bool),
            to(self.dones[idx], torch.bool),
        )

    def __len__(self) -> int:
        return self.size
