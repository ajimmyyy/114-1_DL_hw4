import gzip
import os
import pickle
import cv2
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import configure_logger
from tqdm import tqdm
from tetris_env import TetrisEnv

torch.set_float32_matmul_precision('medium')

N_ENVS = 128
LOG_DIR = "./tetris_tensorboard_logs/"
os.makedirs(LOG_DIR, exist_ok=True)
DEMO_FILE = "tetris_demo_stacked.pkl.gz"

class ViewGameCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        obs = self.locals['new_obs']

        env_0_obs = obs[0, -1, :, :] 

        if isinstance(env_0_obs, torch.Tensor):
            img = env_0_obs.cpu().numpy()
        else:
            img = env_0_obs

        if np.max(img) <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        img_display = cv2.resize(img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Training Preview (Env 0)', img_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        return True

    def _on_training_end(self) -> None:
        cv2.destroyAllWindows()

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

def pretrain_agent(model, epochs=1000):
    if model.replay_buffer.size() < model.batch_size:
        print("Buffer too small for pre-training. Skipping.")
        return

    if model._logger is None:
        tmp_logger = configure_logger(
            model.verbose, 
            model.tensorboard_log, 
            "dqn_pretrain", 
            model.device
        )
        model.set_logger(tmp_logger)
    
    model._current_progress_remaining = 1.0 

    print(f"Starting Offline Pre-training for {epochs} steps...")
    model.train(gradient_steps=epochs, batch_size=model.batch_size)
    
    print("Pre-training finished.")

def main(model_path = None):
    env = make_vec_env(TetrisEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    
    env = VecMonitor(env)
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)  

    eval_env = make_vec_env(TetrisEnv, n_envs=1)
    eval_env = VecMonitor(eval_env)
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_dqn",
        log_path=LOG_DIR,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    if model_path != None and os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        model = DQN.load(model_path, env=env, device="cuda", tensorboard_log=LOG_DIR)
        model.learning_starts = 0
    else:
        model = DQN(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            learning_rate=1e-4,
            
            buffer_size=400_000,  # 經驗回放池大小。
                                # 默認是 1,000,000 (100萬)。
                                # 對於 200x100 圖像，100萬張會吃掉 ~80GB RAM。
                                # 設為 50,000 大約佔用 4GB RAM。如果電腦內存大，可嘗試 100,000。
            
            optimize_memory_usage=True, # 啟用這個！可以大幅減少 RAM 佔用 (不重複存儲 next_obs)
            replay_buffer_kwargs={"handle_timeout_termination": False},
            
            batch_size=1024,      # 每次從 buffer 取多少張圖來學
            learning_starts=10000, # 先玩 10000 步存數據，不訓練，讓 buffer 累積一點多樣性
            target_update_interval=1000, # 每 1000 步更新一次目標網絡 (Target Network)
            train_freq=4,        # 每玩 4 步，訓練一次 (標準配置)
            gradient_steps=1,    # 每次訓練做幾次梯度下降
            
            exploration_fraction=0.3, # 在前 30% 的訓練時間裡，從 100% 隨機慢慢降到 1% 隨機
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02, # 最後保留 2% 的隨機探索
            
            tensorboard_log=LOG_DIR,
            device="cuda"
        )

    print("-------------------------------------------------")
    print(f"Start Online Training...")
    print(f"Final Obs Shape: {env.observation_space.shape}")
    print(f"Replay Buffer Size: {model.buffer_size}")
    
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
        model.save("tetris_dqn")
        print("Model saved as 'tetris_dqn.zip'")
    finally:
        env.close()
    
    model.save("tetris_dqn")
    print("Model saved as 'tetris_dqn.zip'")

if __name__ == "__main__":
    main()