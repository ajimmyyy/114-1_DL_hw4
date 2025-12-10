import math
from typing import Callable
from stable_baselines3.common.vec_env import (
    VecTransposeImage, VecFrameStack, VecMonitor, SubprocVecEnv, VecNormalize
)
from stable_baselines3.common.env_util import make_vec_env
import re
from .client.tetris_env import TetrisEnv
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
from .ddqn import DDQN
import os
from .utils.tetris_cnn import TetrisCNN
from .utils.per import PrioritizedReplayBuffer

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
    "DDQN": DDQN,
}

def create_model(cfg, env, log_dir="./tensorboard_logs", model_path=None):
    algo = cfg["algo"].upper()

    if algo not in ALGOS:
        raise ValueError(f"Unknown algo {algo}")

    AlgoClass = ALGOS[algo]
    params = cfg.get("model", {}).copy()

    use_custom_cnn = params.pop("use_tetris_cnn", False)
    features_dim = params.pop("features_dim", 256)
    use_per = params.pop("use_per", False)
    per_alpha = params.pop("per_alpha", 0.6)
    per_beta = params.pop("per_beta", 0.4)

    params["env"] = env
    params["tensorboard_log"] = log_dir

    if "learning_rate" in params and isinstance(params["learning_rate"], str):
        lr_input = params["learning_rate"]
        match = re.match(r"(\w+)_schedule\(([\d\.e-]+)\)", lr_input)
        
        if match:
            sched_type = match.group(1)
            initial_lr = float(match.group(2))
            
            try:
                params["learning_rate"] = make_learning_schedule(initial_lr, schedule_type=sched_type)
                print(f"[ModelFactory] Using {sched_type} schedule, starting from {initial_lr}")
            except ValueError as e:
                raise ValueError(f"Invalid schedule type in config: {lr_input}") from e
        else:
            try:
                params["learning_rate"] = float(lr_input)
            except ValueError:
                raise ValueError(f"Invalid learning_rate format: {lr_input}")

    if use_custom_cnn:
        params["policy_kwargs"] = dict(
            features_extractor_class=TetrisCNN,
            features_extractor_kwargs=dict(features_dim=features_dim, frame_stack=cfg["env"]["frame_stack"])
        )
    
    if use_per:
        params["replay_buffer_class"] = PrioritizedReplayBuffer
        params["replay_buffer_kwargs"] = dict(alpha=per_alpha, beta=per_beta)

    if "replay_buffer_kwargs" in params:
        rb_kwargs = params["replay_buffer_kwargs"]
        if rb_kwargs.get("handle_timeout_termination", True) and params.get("optimize_memory_usage", False):
            rb_kwargs["handle_timeout_termination"] = False

    if model_path and os.path.exists(model_path):
        print(f"[ModelFactory] Loading model from {model_path}")
        return AlgoClass.load(model_path, env=env, tensorboard_log=params["tensorboard_log"])

    print(f"[ModelFactory] Creating new {algo}")
    return AlgoClass(**params)
    
def make_training_env(n_envs, frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    env = VecNormalize(
        env, 
        norm_obs=False, 
        norm_reward=True, 
        clip_reward=10.0,
        gamma=0.99
    )
    return env

def make_eval_env(frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=1)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    return env

def make_infer_env(frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=1, env_kwargs={"reward_type": "eval"})
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    return env

def make_learning_schedule(initial_value: float, schedule_type: str = "linear") -> Callable[[float], float]:
    if schedule_type == "linear":
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    elif schedule_type == "constant":
        def func(progress_remaining: float) -> float:
            return initial_value
        return func

    elif schedule_type == "exponential":
        def func(progress_remaining: float) -> float:
            return initial_value * (progress_remaining ** 2)
        return func

    elif schedule_type == "cosine":
        def func(progress_remaining: float) -> float:
            return 0.5 * initial_value * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return func

    elif schedule_type == "step":
        def func(progress_remaining: float) -> float:
            if progress_remaining < 0.5:
                return initial_value * 0.1
            return initial_value
        return func

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")