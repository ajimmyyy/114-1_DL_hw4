import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./tensorboard_logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--use_eval", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--view_game", action="store_true")
    parser.add_argument("--save_verbose_log", action="store_true")
    return parser.parse_args()
