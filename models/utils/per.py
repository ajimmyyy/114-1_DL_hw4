import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, *args, alpha=0.6, beta=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    def add(self, *args, **kwargs):
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.priorities.max() if self.size > 0 else 1.0

    def sample(self, batch_size, env=None):
        if self.size == 0:
            raise ValueError("Replay buffer is empty")

        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = super().sample(batch_size, env)

        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        samples.extras = {"weights": torch.tensor(weights, dtype=torch.float32), "indices": indices}
        return samples

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors) + 1e-6
