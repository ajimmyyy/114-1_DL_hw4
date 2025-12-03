from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tetris_env import TetrisEnv

env = TetrisEnv()
check_env(env)

model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
)

model.learn(total_timesteps=500000)

model.save("tetris_ppo")