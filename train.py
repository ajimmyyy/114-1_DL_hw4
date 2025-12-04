import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
from tqdm import tqdm
from tetris_env import TetrisEnv

N_ENVS = 16

LOG_DIR = "./tetris_tensorboard_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            tqdm.write(f"t={self.num_timesteps} | last_reward={np.mean(self.locals.get('rewards'))}")
        return True

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        
    env = make_vec_env(
        TetrisEnv, 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv
    )

    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        batch_size=1024,
        n_steps=1024,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        device="cuda",
    )

    print("-------------------------------------------------")
    print(f"Start Training with {N_ENVS} parallel environments...")
    print("Press Ctrl+C to stop safely.")
    print(f"Monitor logs will be saved to: {LOG_DIR}")
    print("-------------------------------------------------")

    try:
        model.learn(
            total_timesteps=1_000_000, 
            progress_bar=True, 
            log_interval=1,
            callback=DebugCallback(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    finally:
        env.close()

    model.save("tetris_ppo")
    print("Model saved as 'tetris_ppo.zip'")

if __name__ == "__main__":
    main()