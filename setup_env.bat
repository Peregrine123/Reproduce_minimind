@echo off
chcp 65001 >nul
echo ========================================
echo    WandB API Key 快速设置工具
echo ========================================
echo.

set /p WANDB_KEY="请粘贴你的 WandB API Key: "

if "%WANDB_KEY%"=="" (
    echo [错误] API Key 不能为空！
    pause
    exit /b 1
)

echo.
echo 正在创建 .env 文件...

(
echo # WandB Configuration
echo WANDB_API_KEY=%WANDB_KEY%
echo.
echo # 可选配置
echo # WANDB_PROJECT=minimind
echo # WANDB_ENTITY=your_team_name
) > .env

if exist .env (
    echo [成功] .env 文件已创建！
    echo.
    echo 文件位置: %CD%\.env
    echo.
    echo 注意：
    echo   - .env 文件已被 Git 忽略，不会上传到远程仓库
    echo   - 请妥善保管你的 API Key
    echo   - 现在可以直接运行训练脚本了
    echo.
) else (
    echo [错误] 创建 .env 文件失败！
    pause
    exit /b 1
)

echo 按任意键退出...
pause >nul
