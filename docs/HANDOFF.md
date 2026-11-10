# 交接：进度、已定的事、下一步（2026-11-10）

云端会话到此为止，之后在用户本机的 Claude Code 会话里继续。分支：`claude/factor-library`（包含之前所有分支的工作，都已推送）。

## 一、目标和约束（详见 `docs/STRATEGY.md`）

- 个人投资，最终实盘；10 万元；只做多、不对冲、不加杠杆，可以持有现金
- 股票池：全 A **主板**（600/601/603/605/000/001/002/003），剔除 ST、上市不满一年、成交额最小 20%
- 持仓 15 只（10 只作对照），跌出前 45 名才卖；单股 ≤ 20%、单行业 ≤ 30%（买入时检查）；个股止损 −20%，20 天内不买回
- 成本：佣金万 2.5、最低 5 元，印花税，过户费 0.001%，滑点 0.1%
- 执行：T 日收盘出信号，T+1 收盘前手动下单（以后再接 QMT/PTrade 自动化）
- 仓位控制：A（中证全指均线过滤）为主，C（账户回撤 20% 清仓）兜底；B（波动率控制）只作对照
- 成功标准（最终测试，2022 年起只跑一次）：① 年化 > 沪深300 全收益（H00300）② 最大回撤 **不超过沪深300 全收益同期最大回撤**（2026-11-10 从 20% 改）③ 成本翻倍仍跑赢 ④ 跑赢同规则的 Alpha158 + LightGBM
- 样本：训练 2010–2018，验证 2019–2021，测试 2022 起（封存）

## 二、已经完成的

| 计划 | 内容 | 结果 |
|---|---|---|
| 1 全 A 数据 | 本机采集 baostock 日线（5505 只）、东方财富季度财务、指数；打包解包 | 数据版本 `local-20261101`；发现并修好 31 只日线截断（`--part check --fix`） |
| 2 账户回测 | `qt/account.py`：按手取整、最低佣金、上限、止损、仓位控制、回撤兜底 | 完成 |
| 3a 因子库 | 500 个因子（Alpha101、Alpha191、Alpha158、价量补充、估值财务规模） | `reports/factor_library/` |
| 3b 单因子检验与筛选 | 全 A 主板训练段逐个检验，八大类（含"短周期价量"）每类最多 3 个 | 59 个通过，选 21 个：`reports/screening/full_a_train_neutral/selected.csv`；405 个被"扣成本后跑输"刷掉 |
| 4 合成与仓位控制 | 进行中，见下 | |

选中的 21 个：价值 ep_ttm、bp、sp_ttm｜低波动 skew60、ivol20、vol60｜动量/反转 intraday20、ret60、low52w｜成长 sue、deduct_yoy、revenue_yoy｜流动性 abn_turn、turn_12m｜短周期价量 gtja097、gtja095、gtja071｜规模 price_level｜质量 gross_margin_ttm、roe_ttm、net_margin_ttm。

## 三、计划 4 现在的状态（`docs/superpowers/plans/2026-11-09-composite-position-control.md`）

用户的选择：21 个全用；八类等权（每类先标准化）；B 做对照；对照基准每天都用来训练。

已完成：任务 1（登记规则）、2（指数数据：中证全指、沪深300 全收益 2009–2021 完整）、3（`qt/composite.py`）、4（`scripts/build_composite.py`，结果在 `data/cache/composite/`）、5（`scripts/run_position_control.py`）、6 的代码（`scripts/run_baseline.py`）。

**第一次回测结果**（`reports/combination/full_a/README.md`）：

| | 训练段年化 / 最大回撤 | 验证段年化 / 最大回撤 |
|---|---|---|
| 不控仓 | 16.1% / −45.7% | 16.4% / −23.4% |
| A 的 6 个组合 | 4.8%–11.1% / −25% ~ −57% | −0.3%–9.8% / −27.6% ~ −36.6% |
| 沪深300 全收益 | 0.2% | 21.8% |

- 按当时规则（两段回撤都 ≤ 20%）没有组合达标 → 停下讨论
- 看到的问题：① 选股在训练段有效，但验证段跑输沪深300（2019–2020 大盘蓝筹领涨，策略偏中小盘、价值、反转）② 均线控仓在验证段来回挨打，回撤反而变大 ③ C 每次清仓后重置最高点，几轮叠加管不住总回撤
- 用户决定：选项 a（先跑对照基准再决定）；回撤上限改成"不超过沪深300 全收益同期最大回撤"（看过验证段之后改的，已在 `DECISIONS.md` 写明）
- 从图上估计验证段沪深300 全收益最大回撤约 −17%，新规则下可能仍然没有组合达标

**正在本机运行 / 待运行**（单进程）：

```bash
conda activate qt && cd ~/qt-full-a
python scripts/run_position_control.py
python scripts/run_baseline.py
git add reports/combination/full_a reports/trials.jsonl
git commit -m "新回撤上限重新挑选；对照基准 Alpha158 + LightGBM"
git push
```

- `run_position_control.py`：按新上限重新挑选，约 15–25 分钟，结果在 `reports/combination/full_a/README.md`
- `run_baseline.py`：逐年算 Alpha158 特征（可断点续跑）、训练 LightGBM（内存约 4–6 GB）、验证段回测，可能 1–3 小时，结果在 `reports/combination/full_a/baseline.md`。被系统结束的话**不要擅自改成抽样训练**，先和用户商量（已登记的规则）

## 四、下一步

1. 两个脚本跑完后，把合成分和对照基准放在一起分析：
   - 对照基准验证段是否也跑输沪深300、回撤也超标？→ 是：问题主要在这段行情和账户条件；否：问题在我们的因子
   - 两边的验证段 RankIC、各仓位控制方案的收益和回撤
2. 和用户讨论（之前列的选项）：b 调整目标；c 回头改进选股（比如大小盘风格切换）。任何改动先写进 `DECISIONS.md`、计入试验次数
3. 如果有组合达标：跑选中组合上的只报告变体（持仓 10 只、去掉短周期价量、成本翻倍），完成任务 7（四条标准预演、DSR）和任务 8（审核节点，用户决定是否进入最终测试）
4. 之后：计划 5 每日出单脚本，计划 6 自动交易

## 五、常用命令和坑

- 跑测试：`python -m pytest -q -m "not slow and not data"`（约 600 个，30 秒）
- 全 A 筛选：`python scripts/run_full_a_screen.py --jobs 1`（断点续跑；`--reselect` 只重新筛选；`--factors a b` 调试）
- 核对基准：`python scripts/check_benchmark.py`；核对指数：`python scripts/check_indexes.py`
- 数据：补采 `python scripts/collect_full_a.py --part daily`，检查 `--part check --fix`，打包 `--part pack`，本机解包 `python scripts/download_full_a.py --archive data/full_a_日期.tar`
- 多进程会被系统结束，用 `--jobs 1`；终端命令里别夹 `#` 注释；推送用 GitHub 个人访问令牌（勾 `repo` 权限）
- "能买股票等权"基准在训练段年化 −9.9%：每天重挑会把已持有、连续涨停的股票剔除，低估了约 9 个点/年；2026-11-10 起诊断另列"能持有等权"（`DATA_NOTES.md`、DECISIONS「基准更正」）

## 六、更新（2026-11-10，本机会话）

- 两个脚本都跑完了，结果已提交：沪深300 全收益同期最大回撤训练段 −46.1%、验证段 −17.1%；A 的 6 个组合验证段 −27.6% ~ −36.6%，仍没有组合达标
- 对照基准：RankIC 0.0888（合成分 0.0736），但日均换手 59.6%、年成本 156%，验证段不控仓年化 −72.8%；零成本时 +32.6%。不是程序错误（核对见 `DECISIONS.md`）
- **用户决定：计划 4 判为验证段不通过，选 c，回到因子研究（计划 3c）**。测试段未动。上面第四节的第 1–3 步作废
- 下一步：和用户讨论 3c 研究什么，写计划，确认后再做
- 调研（2026-11-10）：计划 4 为什么差、3c 候选方向，见 `docs/research/2026-11-10-why-plan4-failed.md`。结论：因子在验证段仍有效（RankIC 0.085 → 0.074），每年都跑赢能买的股票；跑输沪深300 主要因为能买的股票 2019–2020 比沪深300 少 24–26 个点（大盘蓝筹行情），其次是每年 6–7% 的成本

- 计划 3c 设计：`docs/superpowers/specs/2026-11-10-plan3c-design.md`；实施计划：`docs/superpowers/plans/2026-11-10-robust-low-turnover.md`（进行中，按任务顺序执行；规则见 DECISIONS 2026-11-10「计划 3c 事先登记的规则」）
- 计划 3c 代码已全部完成并通过复查（2026-11-10）。接下来按顺序在本机运行（单进程），每步结果提交推送后再跑下一步：
  1. `python scripts/build_pool.py`（约 1–2 小时；退出码 2 = 高价股买不起比例超标，停下讨论）
  2. `python scripts/run_plan3c_train.py`（约 4–7 小时，可断点续跑；退出码 1 = 某个检查没过，看 `reports/plan3c/train/README.md`）
  3. 只有第 2 步写出 `reports/plan3c/train/final.json` 才跑：`python scripts/run_plan3c_valid.py`（只能跑一次）
- 进度（2026-11-10）：第 1 步通过（买不起比例最高 3.6%）；第 2 步所有检查通过，方案固定为 turn_12m + ep_deduct、15 只、每 10 天调仓、跌出前 45 名才卖、行业上限 20%（`reports/plan3c/train/README.md`）。训练段真实账户年化 19.1%、最大回撤 −24.7%；不做回撤控制时 10.1%、−65%，收益约一半来自回撤分档
- 跑验证段前做了基准更正（只改报告里的说明数字）：诊断加"能持有等权"、每年平均持仓比例、空仓天数（DECISIONS 2026-11-10「基准更正」）
- 下一步：第 3 步 `python scripts/run_plan3c_valid.py`（只能跑一次），之后任务 11 审核节点
- **计划 3c 验证段已跑（2026-11-10）：不通过**（①②③④ 不满足，⑤ 满足；DSR 0.007）。回撤分档在验证段是主要拖累（不做回撤控制 25.8% vs 13.0%）；更正后的"能持有等权"验证段 21.8% vs 22.0%，和沪深300 几乎一样。详见 DECISIONS 2026-11-10「计划 3c 验证段结果」。验证段已用完（看过三次），测试段仍封存。下一步：和用户讨论方向
