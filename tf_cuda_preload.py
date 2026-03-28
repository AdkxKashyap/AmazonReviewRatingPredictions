from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _candidate_nvidia_roots() -> list[Path]:
    roots: list[Path] = []

    for python_lib in (Path(sys.prefix) / "lib").glob("python*/site-packages/nvidia"):
        if python_lib.is_dir():
            roots.append(python_lib)

    repo_venv = Path(__file__).resolve().parent / "venv"
    for python_lib in (repo_venv / "lib").glob("python*/site-packages/nvidia"):
        if python_lib.is_dir():
            roots.append(python_lib)

    return roots


def preload_tensorflow_cuda_libraries() -> None:
    if os.name != "posix":
        return

    for nvidia_root in _candidate_nvidia_roots():
        for lib_dir in [
            nvidia_root / "cuda_runtime/lib",
            nvidia_root / "cublas/lib",
            nvidia_root / "cudnn/lib",
            nvidia_root / "cufft/lib",
            nvidia_root / "curand/lib",
            nvidia_root / "cusolver/lib",
            nvidia_root / "cusparse/lib",
            nvidia_root / "cuda_cupti/lib",
            nvidia_root / "cuda_nvrtc/lib",
            nvidia_root / "nccl/lib",
            nvidia_root / "nvjitlink/lib",
        ]:
            if not lib_dir.is_dir():
                continue

            for library_path in sorted(lib_dir.glob("*.so*")):
                try:
                    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue


preload_tensorflow_cuda_libraries()