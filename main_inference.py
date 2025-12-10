import argparse
import time
import sys

from configs.utils.config_loader import load_config
from models.factory import create_model, make_infer_env

def parse_args():
    parser = argparse.ArgumentParser(description="Tetris RL Inference Script")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the config file (e.g., configs/dqn.yaml)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (e.g., logs/best_model.zip)")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    parser.add_argument("--speed", type=float, default=0.001, help="Delay between steps (seconds)")
    parser.add_argument("--round", type=int, default=5, help="Number of rounds to play (default: infinite until interrupted)")
    return parser.parse_args()

def main(args):
    cfg = load_config(args.cfg)
    print(f"Loaded config: {args.cfg}")

    print("Creating environment...")
    env = make_infer_env(frame_stack=cfg["env"]["frame_stack"])

    if not args.no_render:
        if hasattr(env, 'unwrapped'):
            env.unwrapped.render_mode = 'human'

    print(f"Loading model via create_model...")
    model = create_model(
        cfg=cfg, 
        env=env, 
        log_dir=None,
        model_path=args.model_path
    )

    print("Starting inference... (Press Ctrl+C to stop)")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    total_reward = 0
    episodes = 0

    try:
        while args.round <= 0 or episodes < args.round:
            action, _states = model.predict(obs, deterministic=True)
            step_res = env.step(action)

            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
                done = terminated or truncated
            else:
                obs, reward, done, info = step_res
            
            total_reward += reward.item()

            if not args.no_render:
                env.render()
                time.sleep(args.speed)

            if done:
                episodes += 1
                print(f"Episode {episodes} Finished. Reward: {total_reward:.6f}")

                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                total_reward = 0

    except KeyboardInterrupt:
        print("\nInference stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)