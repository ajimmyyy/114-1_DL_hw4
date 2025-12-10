from stable_baselines3.common.vec_env import (
    VecTransposeImage, VecFrameStack, VecMonitor, SubprocVecEnv
)
from stable_baselines3.common.env_util import make_vec_env
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
    return env

def make_eval_env(frame_stack):
    env = make_vec_env(TetrisEnv, n_envs=1)
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=frame_stack)
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