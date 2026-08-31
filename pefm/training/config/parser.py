import os
from omegaconf import OmegaConf

def merge_yaml_and_args(yaml_path, args):
    if not yaml_path or not os.path.exists(yaml_path):
        return args

    yaml_dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True) or {}
    for key, value in yaml_dict.items():
        setattr(args, key, value)
    return args
