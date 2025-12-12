import argparse
import sys
import os
import shutil
import glob
import time
import numpy as np
import cv2
import imageio
import csv

from configs.utils.config_loader import load_config
from models.factory import create_model, make_infer_env

def parse_args():
    parser = argparse.ArgumentParser(description="Tetris RL Testing & Replay Generation Script")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file (e.g., configs/dqn.yaml)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (e.g., logs/best_model.zip)")
    parser.add_argument("--test_steps", type=int, default=1000, help="Total steps to test (default: 2000)")
    parser.add_argument("--output_dir", type=str, default="./replay", help="Directory to save replay images")
    parser.add_argument("--no_gif", action="store_true", help="Do not generate GIF")
    return parser.parse_args()

def main(args):
    cfg = load_config(args.cfg)
    print(f"Loaded config: {args.cfg}")

    replay_folder = args.output_dir
    if os.path.exists(replay_folder):
        print(f"Cleaning up existing replay folder: {replay_folder}")
        shutil.rmtree(replay_folder)
    os.makedirs(replay_folder, exist_ok=True)

    print("Creating environment...")
    env = make_infer_env(frame_stack=cfg["env"]["frame_stack"])

    print(f"Loading model via create_model...")
    model = create_model(
        cfg=cfg, 
        env=env, 
        log_dir=None,
        model_path=args.model_path
    )

    print(f"Starting testing for {args.test_steps} steps...")
    
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    ep_id = 0
    ep_steps = 0
    current_ep_reward = 0
    
    max_reward = -1e10
    best_ep_id = 0
    max_rm_lines = 0
    max_lifetime = 0
    
    current_ep_folder = os.path.join(replay_folder, str(ep_id))
    os.makedirs(current_ep_folder, exist_ok=True)

    try:
        for step in range(args.test_steps):
            action, _states = model.predict(obs, deterministic=True)

            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs, reward, done, info = step_res

            r = reward.item() if hasattr(reward, 'item') else reward
            current_ep_reward += r

            img_to_save = obs[0]
            if obs.shape[-1] > 3: 
                img_to_save = obs[0][:, :, -3:] 

            fname = os.path.join(current_ep_folder, '{:06d}.png'.format(ep_steps))

            if isinstance(img_to_save, np.ndarray):
                cv2.imwrite(fname, img_to_save)
            
            ep_steps += 1
            
            if (step + 1) % 100 == 0:
                print(f"Step {step+1}/{args.test_steps} | Current Ep Reward: {current_ep_reward:.2f}")

            if done:
                print(f"Episode {ep_id} Finished. Reward: {current_ep_reward:.2f}, Lines: {info[0].get('lines_cleared', 0)}")
                
                if current_ep_reward > max_reward:
                    max_reward = current_ep_reward
                    best_ep_id = ep_id
                    max_rm_lines = info[0].get('lines_cleared', 0)
                    max_lifetime = info[0].get('steps', 0)
                    print(f" -> New Best Episode! (ID: {best_ep_id})")

                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                ep_id += 1
                ep_steps = 0
                current_ep_reward = 0

                current_ep_folder = os.path.join(replay_folder, str(ep_id))
                os.makedirs(current_ep_folder, exist_ok=True)

    except KeyboardInterrupt:
        print("\nTesting interrupted.")
    finally:
        env.close()

    print("-" * 30)
    print("Testing Complete.")
    print(f"Max Reward: {max_reward}")
    print(f"Best Episode ID: {best_ep_id}")
    print(f"Max Removed Lines: {max_rm_lines}")
    print(f"Max Lifetime: {max_lifetime}")

    if not args.no_gif and max_reward > -1e10:
        print("Generating GIF for the best episode...")
        best_replay_path = os.path.join(replay_folder, str(best_ep_id))
        gif_filename = os.path.join(args.output_dir, f'best_replay_ep{best_ep_id}.gif')
        
        filenames = sorted(glob.glob(os.path.join(best_replay_path, '*.png')))
        
        if filenames:
            images = []
            for filename in filenames:
                images.append(imageio.v2.imread(filename))
            
            imageio.mimsave(gif_filename, images, loop=0)
            print(f"GIF saved to: {gif_filename}")
        else:
            print("No images found for the best episode to generate GIF.")

    csv_path = 'submission.csv'
    print(f"Saving results to {csv_path}...")
    with open(csv_path, 'w') as fs:
        fs.write('id,removed_lines,played_steps\n')
        fs.write(f'0,{max_rm_lines},{max_lifetime}\n')
        fs.write(f'1,{max_rm_lines},{max_lifetime}\n')

if __name__ == "__main__":
    args = parse_args()
    main(args)