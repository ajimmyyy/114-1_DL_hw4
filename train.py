import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from tetris_env import TetrisEnv

LOG_DIR = "./tetris_tensorboard_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

env = TetrisEnv()
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env)

model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    n_steps=2048,
    ent_coef=0.01,
    tensorboard_log=LOG_DIR,
)

print("-------------------------------------------------")
print("Start Training... Press Ctrl+C to stop safely.")
print(f"Monitor logs will be saved to: {LOG_DIR}")
print("-------------------------------------------------")

try:
    model.learn(total_timesteps=1_000_000, progress_bar=True, log_interval=1)
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current model...")

model.save("tetris_ppo")
print("Model saved as 'tetris_ppo.zip'")