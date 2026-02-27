# src/utils_gpu.py
import gc

def free_vram():
    """
    Aggressively free Python references + CUDA cache.
    Use this between loading big models (e.g., LLaVA and Mistral).
    """
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()