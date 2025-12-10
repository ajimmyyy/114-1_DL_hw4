import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy


class DDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class DDQNCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DDQN(DQN):
    policy_aliases = {"MlpPolicy": DDQNPolicy, "CnnPolicy": DDQNCnnPolicy}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ---- override TD error ---- #
    def _compute_q_target(self, next_obs, rewards, dones):
        # Q-online(s', a)
        with th.no_grad():
            q_next_online = self.q_net(next_obs)
            next_actions = th.argmax(q_next_online, dim=1)

            # Q-target(s', a*)
            q_next_target = self.target_q_net(next_obs)
            q_values_target = q_next_target.gather(1, next_actions.long().unsqueeze(1)).squeeze(1)

            # TD target
            q_target = rewards + (1 - dones) * self.gamma * q_values_target

        return q_target

    # override DQN._training_step (most stable way)
    def _training_step(self, gradient_steps: int, batch_size: int = 100):
        for _ in range(gradient_steps):

            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Q(s, a)
            q_values = self.q_net(replay_data.observations)
            q_values = q_values.gather(1, replay_data.actions.long().unsqueeze(1)).squeeze(1)

            # ---- Double DQN TD-Target ---- #
            q_target = self._compute_q_target(
                replay_data.next_observations,
                replay_data.rewards,
                replay_data.dones,
            )

            # Loss
            loss = nn.functional.smooth_l1_loss(q_values, q_target)

            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        return loss.item()
