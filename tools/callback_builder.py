import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from .callbacks import ViewGameCallback, DebugCallback, SaveVerboseLogCallback

def build_callbacks(args, cfg, env):
    callbacks = []

    checkpoint = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(args.save_dir, f"{cfg['cfg_name']}"),
        name_prefix="rl_model"
    )
    callbacks.append(checkpoint)

    if args.use_eval:
        eval_callback = EvalCallback(
            env,
            best_model_save_path=args.save_dir,
            log_path=args.save_dir,
            eval_freq=50000,
            deterministic=True
        )
        callbacks.append(eval_callback)

    if args.view_game:
        view_game_callback = ViewGameCallback()
        callbacks.append(view_game_callback)
    if args.debug:
        debug_callback = DebugCallback()
        callbacks.append(debug_callback)
    if args.save_verbose_log:
        save_verbose_log_callback = SaveVerboseLogCallback(
            log_path=os.path.join(args.save_dir, "verbose_log.csv")
        )
        callbacks.append(save_verbose_log_callback)

    return callbacks
