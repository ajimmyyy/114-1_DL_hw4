import cv2
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class ViewGameCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            obs = self.locals['new_obs']

            if hasattr(self.training_env, "unnormalize_obs"):
                obs = self.training_env.unnormalize_obs(obs)
                
            frame = obs[0, -1, :, :]

            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()

            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
                
            img = frame
            img_display = cv2.resize(
                img, (0, 0),
                fx=4, fy=4,
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow('Training Preview (Env 0)', img_display)

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