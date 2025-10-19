# MiniMind 工程化审查文档

## 🎯 关键行动

1. 建立统一的 `src/` 包结构，拆分 CLI、训练循环、数据管线与模型定义，彻底移除运行期 `exec`。
2. 引入集中化配置系统（推荐 `Hydra` 或 YAML + `argparse`），抽离当前散落的超参数与路径，提升可复用性与可调参能力。
3. 重写 `dataset/lm_dataset.py` 并补充测试用例，修复 loss mask、角色名和方法引用等错误，确保训练输入可信。

## 📐 架构与模块化

- **问题描述**：`main.py` 在 CLI 中直接 `exec` 训练脚本（`main.py:195-205`），`triainer/` 内的函数通过全局变量共享状态（`triainer/train_pretrian.py:35-101`），形成双入口和紧耦合；缺乏 `src/` 包结构，代码更像脚本集合。
- **改进建议**：规划 `src/minimind/` 包结构，内含 `cli.py`、`training/`、`data/`、`models/`、`utils/`；主入口只导入公共 API；训练循环封装为可复用模块，避免在脚本中共享全局状态。

## 🗃 配置管理

- **问题描述**：关键超参和路径硬编码且相互矛盾，例如 Kaggle 路径（`triainer/train_pretrian.py:172`）与主 CLI 默认不一致；MoE、学习率等在多个脚本重复定义。
- **改进建议**：引入集中配置（推荐 Hydra）：使用 `conf/{task}.yaml` 管理模型、优化器、数据参数；CLI 支持 `--config-path`/`--config-name`；移除散落的默认值，仅在配置层维护。

## 🌱 可复现性

- **问题描述**：仅设置 `torch.manual_seed` 与 `torch.cuda.manual_seed`，未同步 `random`、`numpy`；缺少 README 描述环境准备；DDP 场景下种子偏移逻辑分散。
- **改进建议**：在统一入口设置所有随机源（`random`、`numpy`、`torch`、`torch.cuda`），在 DDP 中对 rank 做偏移；在 README 说明 `uv sync`、命令、环境变量；提供脚本或 `Makefile` 以帮助复现实验。

## 💾 数据管道

- **问题描述**：`dataset/lm_dataset.py` 存在多处错误：返回未定义变量 `loss_mask`（`dataset/lm_dataset.py:98`）、角色拼写 `assisant`（`dataset/lm_dataset.py:71`）、DPO 数据集调用不存在的 `_generation_loss_mask`（`dataset/lm_dataset.py:163`）；`apply_chat_template` 误写为 `tokenizer=False`（`dataset/lm_dataset.py:141`）；`DataLoader` 配置不可配置。
- **改进建议**：按任务拆分数据集至 `data/` 模块，重写 loss mask 与 prompt 逻辑并添加单元测试；数据路径由配置提供；`DataLoader` 的 `shuffle`、`num_workers`、`pin_memory` 根据场景或配置调整。

## 🧠 模型实现

- **问题描述**：类名 `MiniMindForCasualLM` 与 SFT 脚本引用 `MiniMindForCausalLM` 不一致（`model/model_minimind.py:439` vs `triainer/train_full_sft.py:15`）；模型输出持久化在 `self.OUT`，不利于并行安全；缺少组件化说明。
- **改进建议**：统一命名并提供 `from_config` 工厂；将 MoE、Attention 等组件拆分到 `models/components/`；`forward` 返回局部构造的输出对象；为关键模块补充 docstring。

## 🔄 训练与评估逻辑

- **问题描述**：训练循环包含调度、日志、保存等所有逻辑，依赖多个全局变量；DDP 初始化散落，SFT 默认 `shuffle=False`；缺乏验证或指标统计。
- **改进建议**：设计 `Trainer` 类或函数式 API，包含 `train_epoch`、`evaluate`、`save_checkpoint` 等模块；封装 DDP 初始化；扩展配置允许启用验证集和评估脚本；根据需求考虑整合 `Accelerate`/`Lightning`。

## 🧼 代码质量与可读性

- **问题描述**：存在乱码注释、拼写错误（`Logger`/`logger`、`assisant`），未用变量（`tokens_per_iter`）、缺少 README；类型注解和 lint 缺失；`MiniMind` 配置缺 docstring。
- **改进建议**：运行 `ruff` 清理未用代码；补充类型标注与文档；修复中文注释编码问题；撰写 `README.md`，说明项目目标、目录、配置、运行与评估流程、常见问题；建立 CI 执行 lint 与单测。

## 建议实施步骤

1. 建立 `src/` 结构并迁移 CLI/训练/模型模块，补充 README 与配置说明。
2. 重写数据集模块并添加单元测试，确保训练输入和 mask 逻辑正确。
3. 重构训练循环与配置系统，实现统一入口、可扩展评估和 DDP/单机共用逻辑。
