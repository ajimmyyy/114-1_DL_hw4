import cv2
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class ViewGameCallback(BaseCallback):
    def __init__(self, frame_num=4, verbose=0):
        super().__init__(verbose)
        self.frame_num = frame_num

    def _on_step(self) -> bool:
        try:
            obs = self.locals['new_obs']

            if hasattr(self.training_env, "unnormalize_obs"):
                obs = self.training_env.unnormalize_obs(obs)

            frame_stack = obs[0]

            start = 3 * (self.frame_num - 1)
            end   = 3 * self.frame_num
            frame = frame_stack[start:end]

            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            frame = np.transpose(frame, (1, 2, 0))

            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

            img_display = cv2.resize(
                frame, (0, 0),
                fx=4, fy=4,
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow("Training Preview (RGB)", img_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

            return True

        except Exception as e:
            print("[ViewGameCallback] Error:", e)
            return True

    def _on_training_end(self) -> None:
        cv2.destroyAllWindows()

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            tqdm.write(f"t={self.num_timesteps} | last_reward={np.mean(self.locals.get('rewards'))}")
        return True
    
class SaveVerboseLogCallback(BaseCallback):
    def __init__(self, log_path="verbose_log.txt", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.f = open(self.log_path, "w")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        try:
            ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
            ep_len_mean = self.logger.name_to_value.get("rollout/ep_len_mean", None)

            if ep_rew_mean is not None and ep_len_mean is not None:
                with open(self.log_path, "a") as f:
                    f.write(f"{self.num_timesteps},{ep_rew_mean},{ep_len_mean}\n")

        except Exception as e:
            print("[SaveVerboseLogCallback] Error:", e)

    def _on_training_end(self):
        print(f"Logs saved to {self.log_path}")
