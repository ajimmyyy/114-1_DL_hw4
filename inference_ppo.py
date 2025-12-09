# inference.py
import os
import time
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv, VecMonitor, VecVideoRecorder, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from tetris_env import TetrisEnv

ALGO = "DQN"
MODEL_PATH = "weight/tetris_dqn_v2.zip"
NUM_ENVS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER = True
RECORD = False
RECORD_FOLDER = "videos"
FRAME_STACK = 4
EPISODES = 3
VIDEO_RECORD_EVERY_EP = 1

def make_single_env():
    return TetrisEnv(reward_type='eval')

def main():
    print(f"Running inference with {ALGO} on {DEVICE}")

    if NUM_ENVS == 1:
        vec_env = DummyVecEnv([make_single_env])
    else:
        vec_env = make_vec_env(TetrisEnv, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)

    vec_env = VecMonitor(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK)

    if RECORD:
        try:
            os.makedirs(RECORD_FOLDER, exist_ok=True)
            vec_env = VecVideoRecorder(
                vec_env,
                RECORD_FOLDER,
                record_video_trigger=lambda x: True,
                video_length=1000,
                name_prefix=f"{ALGO}_eval",
            )
        except ModuleNotFoundError:
            print("moviepy not installed. Skipping video recording.")

    if ALGO == "PPO":
        model = PPO.load(MODEL_PATH, device=DEVICE, env=vec_env)
    elif ALGO == "DQN":
        model = DQN.load(MODEL_PATH, device=DEVICE, env=vec_env)
    else:
        raise ValueError("ALGO must be 'PPO' or 'DQN'")

    model.policy.to(DEVICE)

    episode_count = 0
    obs = vec_env.reset()
    episode_rewards = np.zeros(NUM_ENVS, dtype=float)
    episode_lengths = np.zeros(NUM_ENVS, dtype=int)

    try:
        while episode_count < EPISODES:
            actions, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)

            if RENDER:
                try:
                    vec_env.render()
                except Exception:
                    pass

            episode_rewards += rewards
            episode_lengths += 1

            for i, done in enumerate(dones):
                if done:
                    info = infos[i] if isinstance(infos, (list, tuple)) else infos
                    ep_info = info.get("episode") if isinstance(info, dict) else None
                    if ep_info is not None:
                        print(f"[Episode {episode_count+1}] reward={ep_info['r']:.2f}, length={ep_info['l']}")
                    else:
                        print(f"[Episode {episode_count+1}] reward={episode_rewards[i]:.2f}, length={episode_lengths[i]}")

                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    episode_count += 1
                    if episode_count >= EPISODES:
                        break

            if RENDER:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Inference interrupted by user.")
    finally:
        vec_env.close()
        print("Inference finished.")

if __name__ == "__main__":
    main()
