#!/bin/bash

echo "========================================"
echo "   WandB API Key 快速设置工具"
echo "========================================"
echo

read -p "请粘贴你的 WandB API Key: " WANDB_KEY

if [ -z "$WANDB_KEY" ]; then
    echo "[错误] API Key 不能为空！"
    exit 1
fi

echo
echo "正在创建 .env 文件..."

cat > .env << EOF
# WandB Configuration
WANDB_API_KEY=$WANDB_KEY

# 可选配置
# WANDB_PROJECT=minimind
# WANDB_ENTITY=your_team_name
EOF

if [ -f .env ]; then
    echo "[成功] .env 文件已创建！"
    echo
    echo "文件位置: $(pwd)/.env"
    echo
    echo "注意："
    echo "  - .env 文件已被 Git 忽略，不会上传到远程仓库"
    echo "  - 请妥善保管你的 API Key"
    echo "  - 现在可以直接运行训练脚本了"
    echo

    # 设置文件权限，只有当前用户可读写
    chmod 600 .env
    echo "  - 已设置 .env 文件权限为 600 (仅当前用户可读写)"
else
    echo "[错误] 创建 .env 文件失败！"
    exit 1
fi
