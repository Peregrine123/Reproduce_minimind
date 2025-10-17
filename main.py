#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind项目主入口文件
提供预训练和SFT训练的统一入口
"""

import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="MiniMind 训练脚本")
    parser.add_argument(
        "mode", 
        choices=["pretrain", "sft"], 
        help="训练模式: pretrain(预训练) 或 sft(监督微调)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="配置文件路径(可选)"
    )
    # 解析已知参数，剩余参数传递给对应的训练脚本
    args, remaining_args = parser.parse_known_args()
    
    if args.mode == "pretrain":
        from triainer.train_pretrian import main as pretrain_main
        # 修改sys.argv以传递剩余参数
        sys.argv = ["train_pretrian.py"] + remaining_args
        pretrain_main()
    elif args.mode == "sft":
        from triainer.train_full_sft import main as sft_main
        # 修改sys.argv以传递剩余参数
        sys.argv = ["train_full_sft.py"] + remaining_args
        sft_main()

if __name__ == "__main__":
    main()