"""可复现性工具模块

提供统一的随机种子设置函数，确保实验可复现。
"""

import random
import numpy as np
import torch


def set_seed(seed: int, rank: int = 0, offset_by_rank: bool = False):
    """设置所有随机数生成器的种子，确保实验可复现

    Args:
        seed: 基础随机种子
        rank: DDP 进程的 rank（用于分布式训练）
        offset_by_rank: 是否根据 rank 偏移种子（DDP 场景下推荐 True）

    Examples:
        # 单机训练
        set_seed(42)

        # DDP 训练（每个进程使用不同的种子）
        set_seed(42, rank=rank, offset_by_rank=True)
    """
    effective_seed = seed + rank if offset_by_rank else seed

    # Python 内置随机模块
    random.seed(effective_seed)

    # NumPy 随机模块
    np.random.seed(effective_seed)

    # PyTorch CPU 随机模块
    torch.manual_seed(effective_seed)

    # PyTorch CUDA 随机模块
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)  # 多 GPU 场景

    # 确保 cuDNN 的确定性行为（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_reproducibility_info():
    """获取当前可复现性相关的环境信息

    Returns:
        dict: 包含 PyTorch、CUDA、cuDNN 版本信息
    """
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

    return info
