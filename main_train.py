from models.args import parse_args
from configs.utils.config_loader import load_config
from models.factory import create_model, make_training_env, make_eval_env
from tools.callback_builder import build_callbacks
from tools.trainer import RLTrainer
import logging

def main(args):
    cfg = load_config(args.cfg)

    env = make_training_env(
        n_envs=cfg["env"]["n_envs"],
        frame_stack=cfg["env"]["frame_stack"]
    )

    eval_env = make_eval_env(frame_stack=cfg["env"]["frame_stack"])

    model = create_model(cfg, env, log_dir=args.log_dir)

    callbacks = build_callbacks(args, cfg, env)

    trainer = RLTrainer(
        model=model,
        env=env,
        eval_env=eval_env,
        cfg=cfg,
        save_dir=args.save_dir,
    )

    trainer.learn(callbacks=callbacks)

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        filename="train_log.txt",
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
    )
    
    main(args)
