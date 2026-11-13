# 给 Claude Code 的工作说明

A 股个人投资量化策略（10 万元、只做多、全 A 主板、最终实盘）。研究在用户自己的 Mac 上进行。
**开始工作前先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)（进度和下一步）和 [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md)（研究怎么做，2026-11-12 起）**，再按需看：

- [`docs/STRATEGY.md`](docs/STRATEGY.md)：策略说明书（账户、股票池、交易规则、仓位控制、成功标准）。和其他文档冲突时以它为准
- [`docs/DECISIONS.md`](docs/DECISIONS.md)：每个研究决定和变更记录（按时间往下追加）
- [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md)：数据口径和核对结论
- [`docs/superpowers/plans/`](docs/superpowers/plans/README.md)：实施计划（README 是总览和状态）

## 和用户沟通

- 用清楚的大白话，不堆术语；结论先说，再说依据
- 要用户做决定时，列出选项和我的建议，说明理由
- 给终端命令时：**命令里不要夹 `#` 注释**（zsh 会把它当参数）；说明写在代码块外面；新开终端要先 `conda activate qt && cd ~/qt-full-a`

## 工作流程

用户 2026-11-12 把研究决策和代码库交给 Claude 负责：方案和代码结构自己定，按 `docs/FRAMEWORK.md` 一圈一圈做；只有最终测试和放宽事先登记的门槛要用户批准。

1. 研究：每一圈先写假设卡（`lab/cards/`）并提交，再运行；工程性的大改动仍先写计划（`docs/superpowers/plans/日期-名字.md`）
2. `qt/` 和 `scripts/` 的代码先写测试（TDD）；写完故意改错几处，确认测试能抓到
3. 每完成一个任务：跑测试 `python -m pytest -q -m "not slow and not data"`，提交并推送到当前分支
4. 任何影响结果的规则，**运行前**写进 `DECISIONS.md`；看过结果再改的，要写明"验证段已看过"
5. 每次检验、每个方案都记试验日志 `reports/trials.jsonl`（`qt/trials.py`），最终测试时算 DSR

## 不能做的事

- **测试段（2022-01-01 起）封存**：取数都经过 `qt/config.check_range()`；只有用户批准的最终测试才用 `QT_UNLOCK_TEST=1`，只跑一次
- 不擅自放宽事先登记的门槛；达不到就停下来和用户讨论
- 不把数据传到 GitHub Release（仓库公开）；`data/` 不进仓库
- 不在提交、PR、代码里写模型名称

## 本机环境

- 项目：`~/qt-full-a`；conda 环境 `qt`（Python 3.12，pyqlib 0.9.7，pandas 3）
- Qlib 数据包：`~/.qlib/qlib_data/cn_data`（版本 2026-10-25）；全 A 补充数据：`~/.qlib/qt_full_a`（版本 local-20261101）
- 原始采集：`data/raw/full_a/`；中间结果：`data/cache/`（screen_full_a、composite、baseline）
- **多进程在这台 Mac 上会被系统结束**（`--jobs 2/4` 都出现过 BrokenProcessPool），长任务一律单进程 `--jobs 1`
- pandas 3：`to_numpy()` 可能返回只读数组，要写就加 `copy=True`
