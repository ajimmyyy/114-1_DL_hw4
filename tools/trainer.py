import os
from stable_baselines3.common.callbacks import EvalCallback

class RLTrainer:
    def __init__(self, model, env, eval_env, cfg, save_dir="./checkpoints"):
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.cfg = cfg
        self.save_dir = save_dir

    def learn(self, callbacks=None):
        total_steps = self.cfg["train"]["total_timesteps"]

        print(f"[Trainer] Training {total_steps} steps…")

        try:
            self.model.learn(
                total_timesteps=total_steps,
                progress_bar=True,
                callback=callbacks
            )
        except KeyboardInterrupt:
            print("\n[Trainer] Training interrupted by user (Ctrl+C). Saving model...")
        finally:
            save_path = os.path.join(self.save_dir, f"{self.cfg['cfg_name']}", "final_model.zip")
            self.model.save(save_path)
            print(f"[Trainer] Model saved to {save_path}")
            self.env.close()
            self.eval_env.close()
            print("[Trainer] Training finished.")