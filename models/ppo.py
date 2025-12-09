import gzip
import os
import pickle
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.data.types import Transitions

import torch
from tqdm import tqdm
from utils.callback import ViewGameCallback
from models.utils.tetris_cnn import TetrisCNN
from tetris_env import TetrisEnv

N_ENVS = 16
LOG_DIR = "./tetris_tensorboard_logs/"
DATA_PATH = "tetris_demo_stacked.pkl.gz"
os.makedirs(LOG_DIR, exist_ok=True)

def inject_human_data(model, filename):
    if not os.path.exists(filename):
        print(f"Warning: Demo file {filename} not found. Skipping injection.")
        return

    print(f"Loading human demonstration from {filename}...")
    with gzip.open(filename, 'rb') as f:
        demo_data = pickle.load(f)

    print(f"Injecting {len(demo_data)} transitions into Replay Buffer...")
    
    n_envs = model.n_envs
    
    for obs, action, reward, next_obs, done in tqdm(demo_data, desc="Injecting"):
        action = np.array(action) 
        if n_envs > 1:
            obs = np.repeat(obs, n_envs, axis=0)
            next_obs = np.repeat(next_obs, n_envs, axis=0)
            action = np.repeat(action, n_envs, axis=0)
            reward = np.repeat(reward, n_envs, axis=0)
            done = np.repeat(done, n_envs, axis=0)

        model.replay_buffer.add(
            obs,      
            next_obs, 
            action, 
            reward, 
            done, 
            [{"broadcast": True} for _ in range(len(reward))]
        )
    
    print(f"Buffer filled! Current size: {model.replay_buffer.size()}")

def get_pre_train_data():
    print(f"Loading demonstrations from {DATA_PATH}...")
    with gzip.open(DATA_PATH, 'rb') as f:
        buffer = pickle.load(f)
    
    # buffer 結構: [(obs, action, reward, next_obs, dones), ...]
    # obs shape: (1, 4, H, W) -> 來自 VecEnv
    # action shape: [int]
    
    # 提取並合併數據
    # np.concatenate 會把 list of (1, 4, H, W) 合併成 (N, 4, H, W)
    obs_all = np.concatenate([item[0] for item in buffer], axis=0)
    
    # 處理動作: 將 list of lists 轉為 flat numpy array
    # item[1] 是 [action_int]，我們需要把它變成純量
    acts_all = np.array([item[1][0] for item in buffer])
    
    # 檢查維度
    print(f"Total transitions: {len(buffer)}")
    print(f"Obs shape: {obs_all.shape}")  # 預期: (N, 4, H, W)
    print(f"Acts shape: {acts_all.shape}") # 預期: (N, )
    
    # 建立 imitation 專用的 Transitions 物件
    transitions = Transitions(
        obs=obs_all,
        acts=acts_all,
        infos=[{}] * len(buffer),
        next_obs=np.zeros_like(obs_all), # BC 不需要 next_obs，補零即可
        dones=np.zeros(len(buffer), dtype=bool) # BC 僅依賴 (s, a)，dones 非必須
    )

    return transitions

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        
    env = make_vec_env(
        TetrisEnv, 
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(
        features_extractor_class=TetrisCNN,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        device="cuda",
    )

    imitation_trajectories = get_pre_train_data()

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=imitation_trajectories,
        policy=model.policy,
        rng=np.random.default_rng(),
    )

    print("Starting BC training...")
    bc_trainer.train(n_epochs=10)
    print("BC Pre-training finished.")

    print("-------------------------------------------------")
    print(f"Start Training with {N_ENVS} parallel environments...")
    print("Press Ctrl+C to stop safely.")
    print(f"Monitor logs will be saved to: {LOG_DIR}")
    print("-------------------------------------------------")

    try:
        model.learn(
            total_timesteps=1_000_000_0, 
            progress_bar=True, 
            log_interval=1,
            callback=ViewGameCallback(),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        model.save("tetris_ppo")
        print("Model saved as 'tetris_ppo.zip'")
    finally:
        env.close()

    model.save("tetris_ppo")
    print("Model saved as 'tetris_ppo.zip'")

if __name__ == "__main__":
    main()