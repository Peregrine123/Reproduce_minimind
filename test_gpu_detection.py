#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试多 GPU 检测逻辑"""

import torch

# 检测 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"检测到 GPU 数量: {num_gpus}")

if num_gpus > 1:
    print(f"✓ 系统有 {num_gpus} 个 GPU，可以自动启用分布式训练")
    for i in range(num_gpus):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
elif num_gpus == 1:
    print(f"✓ 系统有 1 个 GPU: {torch.cuda.get_device_name(0)}")
    print("  将使用单 GPU 训练模式")
else:
    print("✗ 未检测到 GPU，将使用 CPU 训练模式")

# 测试 CUDA 是否可用
print(f"\nCUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
