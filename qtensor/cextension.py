# code is based the implementation of "bitsandbytes"
# see https://github.com/TimDettmers/bitsandbytes/tree/main/bitsandbytes

import torch
import platform
import ctypes as ct
from pathlib import Path


DYNAMIC_LIBRARY_SUFFIX = {"Windows": ".dll", "Linux": ".so"}.get(platform.system())


def get_cuda_version():
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        raise RuntimeError("CUDA is not available in this PyTorch build.")
    major, minor = map(int, cuda_version.split("."))
    return f"{major}{minor}"


def get_lib():
    cuda_version_string = get_cuda_version()
    if DYNAMIC_LIBRARY_SUFFIX is None:
        raise RuntimeError("The CUDA quantization library is only distributed for Linux and Windows.")
    binary_name = f"libqtensor_cuda{cuda_version_string}{DYNAMIC_LIBRARY_SUFFIX}"
    binary_path = (Path(__file__).parent / binary_name).resolve()
    if not binary_path.is_file():
        raise FileNotFoundError(f"Quantization library not found at {binary_path}.")
    return ct.cdll.LoadLibrary(str(binary_path).replace("\\", "/"))
