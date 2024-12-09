# ==============================================================================
# Copyright (c) Intel [2024]
# ==============================================================================

from functools import lru_cache
import os

import torch
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_npu_available,
)

@lru_cache(maxsize=None)
def is_torch_hpu_available() -> bool:
    try:
        import habana_frameworks.torch.core
    except ImportError:
        return False
    
    from habana_frameworks.torch.hpu import is_bf16_supported as is_torch_bf16_gpu_available
    
    return torch.hpu.is_available()


_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available() 
if is_torch_hpu_available():
    import habana_frameworks.torch.utils.experimental as htexp
    _is_fp16_available = htexp._is_fp16_supported()

def get_current_device() -> "torch.device":
    r"""
    Gets the current available device.
    """
    if is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_hpu_available():
        device = "hpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    elif is_torch_hpu_available():
        return torch.hpu.device_count()
    else:
        return 0

def is_gpu_or_npu_available() -> bool:
    r"""
    Checks if the GPU/NPU/HPU/XPU is available.
    """
    return is_torch_npu_available() or is_torch_cuda_available() or is_torch_hpu_available()
