# Quantitative-Trading

A 股短期选股因子研究：按"因子挖掘 → 单因子检验 → 筛选 → 合成 → 回测"的流程，用 [Qlib](https://github.com/microsoft/qlib) 在沪深300、中证800、全 A 股上逐步验证。

> 仅用于研究与学习，不构成投资建议。

研究里的每个决定（样本切分、持有期、成本、筛选门槛……）都记在 [`docs/DECISIONS.md`](docs/DECISIONS.md)；数据检查结论在 [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md)。

## 环境

需要 Python 3.12（Qlib 0.9.7 的 wheel 最高到 3.12）。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据

行情来自 [chenditc/investment_data](https://github.com/chenditc/investment_data) 每日发布的 Qlib 数据包（约 570MB），默认解压到 `~/.qlib/qlib_data/cn_data`：

```bash
python scripts/download_data.py            # 最新版
python scripts/download_data.py --tag 2026-10-25
python scripts/check_data.py               # 检查数据，写出 reports/data_check.md
```

换数据目录用环境变量 `QT_DATA_DIR`；外部数据目录和试验日志也可以分别用 `QT_EXTERNAL_DIR`、`QT_TRIALS_FILE` 改到别处（比如拿假数据测试时，不污染正式日志）。

## 单因子检验

```bash
python scripts/run_single_factor.py                      # 沪深300 训练段，全部文献因子
python scripts/run_single_factor.py --factors rev20 vol20
```

报告写到 `reports/single_factor/<股票池>_<样本段>/`，`README.md` 是汇总表，每个因子一份详细报告。每次检验都会记一条到试验日志 `reports/trials.jsonl`。

### 行业和市值中性化

需要申万行业和巨潮股本数据，要在能访问这两个网站的电脑上采集，步骤见 [`data/external/README.md`](data/external/README.md)。数据就位后：

```bash
python scripts/run_single_factor.py --neutralize   # 写到 reports/single_factor/<股票池>_<样本段>_neutral/
```

### Alpha158 与筛选

```bash
python scripts/run_single_factor.py --neutralize --set alpha158 --primary-only --details passed --jobs 4
python scripts/screen_factors.py   # 合并文献因子和 Alpha158 的结果，按相关性去冗余
```

Alpha158 的结果写到 `reports/single_factor/<股票池>_<样本段>_neutral_alpha158/`，只给通过的因子写详细报告；筛选结果写到 `reports/screening/`，最终名单是其中的 `selected.csv`。

## 因子合成与回测

```bash
python scripts/run_combination.py
```

读筛选后的最终名单，训练段估权重，验证段上按"等权 → ICIR → 最大化 ICIR → 线性回归 → LightGBM"逐级比较，再试信号平滑和排雷比例，跑对照基准 Alpha158 + LightGBM 和 Qlib 正式回测（`qt/backtest.py` 里的缓冲规则策略），最后算通缩夏普比率。结果写到 `reports/combination/<股票池>_valid/`。因子分数缓存在 `data/cache/`（不进仓库），换了数据或代码要重算时加 `--refresh`。

## 最终测试段封存

最终测试段（2022 年至今）在研究流程走完之前不能读取。`qt/config.py` 里的 `check_range()` 会拦住任何碰到测试段的取数请求。只有最后做一次最终测试时，才设置 `QT_UNLOCK_TEST=1`。

## 测试

```bash
pytest -q                 # 全部
pytest -q -m "not data"   # 不需要数据包的部分
```

## 目录

```text
qt/          研究代码（配置、取数、因子、检验、中性化、合成、回测……）
scripts/     下载数据、检查数据等命令行工具
tests/       测试
docs/        决策记录、数据笔记
reports/     生成的报告
```
