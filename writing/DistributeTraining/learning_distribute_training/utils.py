import os
import gc
import json
import torch
import psutil
import GPUtil
from collections import defaultdict
from typing import Any, Dict, Optional, Literal

from torch.cuda import device


def write_json(json_path: str, json_file: Dict[str, Any], special_name: Optional[str] = None) -> None:
    if special_name:
        dir_name = os.path.dirname(json_path)
        base_name = os.path.basename(json_path)
        new_name = f"{special_name}_{base_name}"
        json_path = os.path.join(dir_name, new_name)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)

def _get_dist_info(accelerator: object = None) -> tuple[Any, Any, Any] | tuple[int, device | device, bool] | tuple[Literal[0], device, Literal[True]]:
    if accelerator is not None:
        return (
            accelerator.local_process_index,
            accelerator.device,
            accelerator.is_main_process,
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.cuda.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
        is_main = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
        return local_rank, device, is_main

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, device, True

def get_gpu_info(accelerator: object = None) -> dict[Any, Any]:
    local_rank, device, is_main = _get_dist_info(accelerator)
    stats = {}

    try:
        gpus = GPUtil.getGPUs()
        if gpus and local_rank < len(gpus):
            gpu = gpus[local_rank]
            prefix = f"Node_Stats/GPU_{local_rank}"
            stats[f"{prefix}/Compute_Load_Pct"] = gpu.load * 100
            stats[f"{prefix}/VRAM_Used_Pct"] = gpu.memoryUtil * 100
            stats[f"{prefix}/Temperature_C"] = gpu.temperature
    except Exception:
        pass

    if torch.cuda.is_available():
        prefix = f"Framework_Stats/GPU_{local_rank}"
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        stats[f"{prefix}/Allocated_MB"] = allocated
        stats[f"{prefix}/Reserved_MB"] = reserved
        stats[f"{prefix}/Peak_Allocated_MB"] = max_allocated

    if is_main:
        stats["Host_Stats/CPU_Usage_Pct"] = psutil.cpu_percent()
        stats["Host_Stats/RAM_Usage_Pct"] = psutil.virtual_memory().percent

    return stats

def clean_cuda_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()