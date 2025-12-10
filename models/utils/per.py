import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Optional, Union, NamedTuple

class PrioritizedReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority):
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        assert n_envs == 1, "PrioritizedReplayBuffer currently only supports n_envs=1"
        self.alpha = alpha
        self.beta = beta  # Initial beta, can be annealed externally or fixed
        self.max_priority = 1.0
        self.tree = SumTree(buffer_size)

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        # Store with max priority to ensure it gets sampled at least once
        self.tree.add(self.max_priority ** self.alpha)

    def sample(self, batch_size: int, env: Optional[any] = None) -> PrioritizedReplayBufferSamples:
        # 1. Sample indices based on priorities
        segment = self.tree.total_priority / batch_size
        batch_inds = []
        tree_idxs = []
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            (tree_idx, p, data_idx) = self.tree.get_leaf(s)
            
            # Protection against out of bounds if tree calculation has float errors
            if data_idx >= self.buffer_size or data_idx >= self.size():
                # Fallback to random sample if tree error (rare)
                data_idx = np.random.randint(0, self.size())
                tree_idx = data_idx + self.buffer_size - 1
                p = self.tree.tree[tree_idx]

            priorities.append(p)
            batch_inds.append(data_idx)
            tree_idxs.append(tree_idx)

        # 2. Calculate Importance Sampling Weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = (self.size() * sampling_probabilities) ** -self.beta
        weights = weights / weights.max()
        
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
        batch_inds_np = np.array(batch_inds)

        # 3. Get standard samples
        samples = super()._get_samples(batch_inds_np, env=env)

        # 4. Return extended samples including weights and indices
        return PrioritizedReplayBufferSamples(
            observations=samples.observations,
            actions=samples.actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            weights=weights_tensor,
            indices=np.array(tree_idxs) # Return tree indices for updating priorities
        )

    def update_priorities(self, tree_idxs, abs_errors):
        abs_errors = abs_errors + 1e-5  # epsilon to avoid 0 priority
        clipped_errors = np.minimum(abs_errors, 1.0)
        ps = np.power(clipped_errors, self.alpha)

        for tree_idx, p in zip(tree_idxs, ps):
            self.tree.update(tree_idx, p)

        self.max_priority = max(self.max_priority, np.max(abs_errors))