#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式训练端口检查和清理工具
"""

import os
import socket
import subprocess
import sys


def find_free_port(start_port=29500, max_attempts=100):
    """查找一个可用的端口

    Args:
        start_port: 起始端口号
        max_attempts: 最大尝试次数

    Returns:
        可用的端口号
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue

    # 如果指定范围内没有可用端口，使用系统分配
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def check_port(port):
    """检查端口是否被占用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True  # 端口可用
    except OSError:
        return False  # 端口被占用


def find_process_using_port(port):
    """查找占用指定端口的进程"""
    try:
        if sys.platform == 'win32':
            # Windows
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    return pid
        else:
            # Linux/Mac
            result = subprocess.run(
                ['lsof', '-i', f':{port}'],
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                pid = parts[1]
                return pid
    except Exception as e:
        print(f"查找进程时出错: {e}")
    return None


def main():
    print("=" * 60)
    print("PyTorch 分布式训练端口检查工具")
    print("=" * 60)

    # 检查默认端口
    default_port = 29500
    print(f"\n1. 检查默认端口 {default_port}...")
    if check_port(default_port):
        print(f"   ✅ 端口 {default_port} 可用")
    else:
        print(f"   ❌ 端口 {default_port} 被占用")
        pid = find_process_using_port(default_port)
        if pid:
            print(f"   占用进程 PID: {pid}")
            print(f"   可以使用以下命令终止进程：")
            if sys.platform == 'win32':
                print(f"     taskkill /PID {pid} /F")
            else:
                print(f"     kill -9 {pid}")

    # 查找可用端口
    print(f"\n2. 查找可用端口...")
    free_port = find_free_port()
    print(f"   ✅ 找到可用端口: {free_port}")

    # 提供使用建议
    print(f"\n3. 使用建议：")
    print(f"   方式 1 - 自动端口（推荐）：")
    print(f"     python main.py --mode pretrain")
    print(f"     (会自动查找可用端口)")
    print(f"")
    print(f"   方式 2 - 手动指定端口：")
    print(f"     torchrun --nproc_per_node=2 --master_port={free_port} main.py --mode pretrain")
    print(f"")
    print(f"   方式 3 - 使用环境变量：")
    print(f"     export MASTER_PORT={free_port}")
    print(f"     torchrun --nproc_per_node=2 main.py --mode pretrain")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
