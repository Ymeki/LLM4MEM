from dataclasses import dataclass

import tyro
@dataclass
class MainArgs:
    data_path: str = "/data"
    name: str = "Music-20"
    # merging
    min_dis: float = 0.5  
    # pruning
    eps: float = 0.78  # d
    lm_model_or_path: str = "all-MiniLM-L12-v2"
    device: str = "cuda"
    seed: int = 3407
    max_seq_length: int = 64
    batch_size: int = 512

def build_main_args():
    return tyro.cli(MainArgs)
