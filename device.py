"""
Device selection: CUDA → Apple MPS (Metal GPU) → CPU.
"""

import sys
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pin_memory_enabled(device):
    return device.type == "cuda"


def dataloader_num_workers():
    if sys.platform == "darwin":
        return 0
    return 2
