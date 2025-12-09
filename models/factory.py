from stable_baselines3.common.vec_env import (
    VecTransposeImage, VecFrameStack, VecMonitor, SubprocVecEnv
)
from stable_baselines3.common.env_util import make_vec_env
from .client.tetris_env import TetrisEnv
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
import os

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}

def create_model(cfg, env, log_dir="./tensorboard_logs", model_path=None):
    algo = cfg["algo"].upper()

    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo}")

    AlgoClass = ALGOS[algo]
    params = cfg.get("model", {})

    params["env"] = env
    params["tensorboard_log"] = log_dir

    if model_path and os.path.exists(model_path):
        print(f"[ModelFactory] Loading model from {model_path}")
        return AlgoClass.load(model_path, env=env, tensorboard_log=params["tensorboard_log"])

    print(f"[ModelFactory] Creating new {algo}")
    return AlgoClass(**params)
    
def make_training_env(n_envs, frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    return env

def make_eval_env(frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=1)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    return env