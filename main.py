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
# 兼容Jupyter环境，其中__file__可能未定义
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 在Jupyter环境中，使用当前工作目录
    current_dir = os.getcwd()
sys.path.append(current_dir)

def main():
    parser = argparse.ArgumentParser(description="MiniMind 训练脚本")
    parser.add_argument(
        "--mode", 
        choices=["pretrain", "sft"], 
        default="pretrain",
        help="训练模式: pretrain(预训练) 或 sft(监督微调)，默认为pretrain"
    )
    
    # 通用参数
    parser.add_argument("--out_dir", type=str, default="./out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if "cuda" in os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind", help="wandb 项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载器工作进程数")
    parser.add_argument("--ddp", action="store_true", help="是否启用分布式训练")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热步数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层层数")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument("--use_moe", default=False, type=bool, help="是否使用混合专家模型")
    
    # 预训练特定参数
    parser.add_argument("--pretrain_data_path", type=str, default="./dataset/pretrain_hq.jsonl", help="预训练数据路径")
    
    # SFT特定参数
    parser.add_argument("--sft_data_path", type=str, default="./dataset/sft_data.jsonl", help="SFT训练数据路径")
    
    # 解析参数
    args = parser.parse_args()
    
    if args.mode == "pretrain":
        # 构建预训练参数列表
        pretrain_args = [
            "train_pretrian.py",
            f"--out_dir={args.out_dir}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--learning_rate={args.learning_rate}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--num_workers={args.num_workers}",
            f"--accumulation_steps={args.accumulation_steps}",
            f"--grad_clip={args.grad_clip}",
            f"--warmup_iters={args.warmup_iters}",
            f"--log_interval={args.log_interval}",
            f"--save_interval={args.save_interval}",
            f"--local_rank={args.local_rank}",
            f"--hidden_size={args.hidden_size}",
            f"--num_hidden_layers={args.num_hidden_layers}",
            f"--max_seq_len={args.max_seq_len}",
            f"--use_moe={args.use_moe}",
            f"--data_path={args.pretrain_data_path}"
        ]
        
        if args.use_wandb:
            pretrain_args.append("--use_wandb")
            pretrain_args.append(f"--wandb_project={args.wandb_project}-Pretrain")
        
        if args.ddp:
            pretrain_args.append("--ddp")
            
        # 修改sys.argv以传递参数
        sys.argv = pretrain_args
        from triainer.train_pretrian import main as pretrain_main
        pretrain_main()
        
    elif args.mode == "sft":
        # 构建SFT参数列表
        sft_args = [
            "train_full_sft.py",
            f"--out_dir={args.out_dir}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--learning_rate={args.learning_rate}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--num_workers={args.num_workers}",
            f"--accumulation_steps={args.accumulation_steps}",
            f"--grad_clip={args.grad_clip}",
            f"--warmup_iters={args.warmup_iters}",
            f"--log_interval={args.log_interval}",
            f"--save_interval={args.save_interval}",
            f"--local_rank={args.local_rank}",
            f"--hidden_size={args.hidden_size}",
            f"--num_hidden_layers={args.num_hidden_layers}",
            f"--max_seq_len={args.max_seq_len}",
            f"--use_moe={args.use_moe}",
            f"--data_path={args.sft_data_path}"
        ]
        
        if args.use_wandb:
            sft_args.append("--use_wandb")
            sft_args.append(f"--wandb_project={args.wandb_project}-SFT")
        
        if args.ddp:
            sft_args.append("--ddp")
            
        # 修改sys.argv以传递参数
        sys.argv = sft_args
        from triainer.train_full_sft import main as sft_main
        sft_main()

if __name__ == "__main__":
    main()