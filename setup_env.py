#!/usr/bin/env python3
"""
WandB API Key 快速设置工具（跨平台）
支持 Windows/Linux/Mac
"""

import os
import sys
from pathlib import Path


def main():
    print("=" * 50)
    print("   WandB API Key 快速设置工具")
    print("=" * 50)
    print()

    # 获取用户输入
    try:
        wandb_key = input("请粘贴你的 WandB API Key: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[取消] 操作已取消")
        sys.exit(0)

    if not wandb_key:
        print("[错误] API Key 不能为空！")
        sys.exit(1)

    print()
    print("正在创建 .env 文件...")

    # 创建 .env 文件内容
    env_content = f"""# WandB Configuration
WANDB_API_KEY={wandb_key}

# 可选配置
# WANDB_PROJECT=minimind
# WANDB_ENTITY=your_team_name
"""

    # 写入文件
    env_file = Path(".env")
    try:
        env_file.write_text(env_content, encoding="utf-8")

        # 在类 Unix 系统上设置文件权限
        if os.name != 'nt':  # 非 Windows 系统
            os.chmod(env_file, 0o600)
            permission_msg = "  - 已设置 .env 文件权限为 600 (仅当前用户可读写)"
        else:
            permission_msg = ""

        print("[成功] .env 文件已创建！")
        print()
        print(f"文件位置: {env_file.absolute()}")
        print()
        print("注意：")
        print("  - .env 文件已被 Git 忽略，不会上传到远程仓库")
        print("  - 请妥善保管你的 API Key")
        print("  - 现在可以直接运行训练脚本了")
        if permission_msg:
            print(permission_msg)
        print()

    except Exception as e:
        print(f"[错误] 创建 .env 文件失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
