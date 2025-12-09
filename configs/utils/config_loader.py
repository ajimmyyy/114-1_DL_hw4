import os
import yaml

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["cfg_name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg
