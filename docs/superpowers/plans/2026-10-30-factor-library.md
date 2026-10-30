# 因子库扩充 实施计划

> **执行方式：** 按 `CLAUDE.md`，用 superpowers:executing-plans 在当前会话里逐个任务执行。

**目标：** 把能用现有数据算出来的公开因子全部纳入因子库（约 510 个），每个因子能在全 A 主板上算出数值。**这个计划不做因子筛选，不算 IC，不记试验日志**；筛选留给后续计划。

**架构：** 现在的因子写成 Qlib 表达式，但 Qlib 表达式只能逐只股票算时间序列，没有横截面排名（Alpha101、Alpha191 大量用到 `rank`），所以新写一套在"日期 × 股票"宽表上运算的算子和公式解析器：

- `qt/ops.py`：时间序列算子（滚动均值、相关、时序排名、线性衰减加权、最高点距今天数……）和横截面算子（排名、标准化、行业中性化），全部向量化
- `qt/formula.py`：把 Alpha101（WorldQuant 写法）和 Alpha191（国泰君安写法）的**原文公式字符串**解析成算子调用，因子以原文公式保存，便于和出处逐字核对
- `qt/library.py`：因子登记表，每个因子记名字、来源、大类、公式或函数、需要的数据
- `qt/panel.py`：从 Qlib 数据包、全 A 补充数据、季度财务拼出宽表输入

**依据：** `docs/STRATEGY.md` 第 7 节（因子大类）、`docs/DATA_NOTES.md`（数据口径）、用户 2026-10-30 的要求：先全部纳入，后筛选。

## 因子来源

| 族 | 数量 | 出处 | 说明 |
|---|---|---|---|
| 文献因子 | 7 | 已有 | 不变 |
| Alpha158 | 158 | Qlib | 已有，不变 |
| Alpha101 | 101 | Kakushadze (2016), *101 Formulaic Alphas*, arXiv:1601.00991 | 公式取自论文正文；`IndClass.sector/industry/subindustry` 用申万一级/二级/三级；`cap` 用流通市值 |
| Alpha191 | 190 | 国泰君安《基于短周期价量特征的多因子选股体系》(2017) | 公式文本取自 GitHub popbo/alphas 的注释（原研报无法直接访问，需逐条核对明显笔误）；`BENCHMARKINDEX` 用沪深300；#30 要 Fama-French 三因子收益，暂缓 |
| 价量补充 | ~30 | 文献与券商研报 | 见下表 |
| 估值、财务、规模 | ~25 | Barra CNE5 描述变量、Liu-Stambaugh-Yuan (2019)、Hou-Xue-Zhang 异象 | 见下表 |

价量补充（只用日线和换手率）：异常换手（20 日 / 250 日，Liu-Stambaugh-Yuan）、1/3/12 个月平均换手（Barra STOM/STOQ/STOA）、换手率波动、Amihud（5/20/60 日）、最大单日收益（5/20 日）、偏度（20/60 日）、相对市场的 beta（252 日）和特质波动率（20/60 日，CAPM 残差）、12-1 月动量、1 个月反转、隔夜收益和日内收益（20 日累计）、52 周新高距离、低价、理想振幅（开源证券：高价日振幅减低价日振幅）、量价背离。

估值、财务、规模（按 `docs/DATA_NOTES.md`：baostock 估值用前一天反推的每股数据乘当天价格；财务按首次公告日次日起可用）：EP、BP、SP、CFP、扣非 EP；ROE、ROA、毛利率、净利率、经营现金流 / 净利润、应计项目、资产负债率、资产增长；营收单季同比、净利润单季同比、扣非净利润同比、标准化预期外盈余（SUE，季节性随机游走）、ROE 变化；对数流通市值、对数总市值（总市值 = 市盈率 × 滚动四季净利润）、市值三次方（Barra NLSIZE）。

## 全局约束

- 输入一律按 PIT：只用当天收盘及以前的数据；财务按首次公告日的下一个交易日起可用；baostock 估值错开一天
- 价格类因子用复权价算，涨跌停、手数仍用未复权价（这里不涉及）
- 公式里的非整数窗口按 Alpha101 论文的约定向下取整
- 因子值方向不做调整、不做去极值和中性化：这些是检验阶段的事
- 不记试验日志：只定义和计算，不看任何收益相关的统计量
- 测试段封存照旧：计算接口经过 `check_range()`

## 重点检查

1. **公式抄错**：Alpha191 的公式文本是第三方转录。→ 任务 4 对每个因子做解析，任务 6 抽 20 个和手工 pandas 计算逐值对比；明显笔误（如 `DELAT`、`COVIANCE`）改正并在登记表注明
2. **横截面算子用到未来**：排名、标准化只能在同一天的股票之间做。→ 任务 1 的测试：改动明天的数据，今天的因子值不变
3. **全 NaN 或常数因子**：公式写错常表现为整列缺失或不变。→ 任务 7 的体检：每个因子的覆盖率、唯一值个数、无穷值比例
4. **算得太慢**：全 A 约 5000 只 × 3000 天，时序排名这类算子不能用 Python 逐窗口循环。→ 任务 1 用向量化实现，任务 7 记录每个因子的耗时
5. **数据缺失被当成 0**：停牌日价格缺失时，滚动窗口要跳过还是置空。→ 约定：窗口内有缺失时结果为 NaN（和 Qlib 一致），测试覆盖

---

### 任务 1：算子

`qt/ops.py`，测试 `tests/test_ops.py`。

- 时间序列：`delay, delta, ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_argmax, ts_argmin, ts_rank, ts_corr, ts_cov, ts_product, decay_linear, sma(n, m), wma, count, sumif, regbeta, regresi, highday, lowday`
- 横截面：`cs_rank, cs_scale, ind_neutralize(x, groups)`
- 逐元素：`signed_power, sign, log, abs`、比较与条件
- 测试：每个算子一个手算小例子；`cs_rank` 不偷看未来；窗口内有 NaN 时为 NaN；全 A 规模（5000 × 3000）下 `ts_rank`、`decay_linear` 各在 10 秒内

### 任务 2：公式解析器

`qt/formula.py`，测试 `tests/test_formula.py`。

- 支持 `+ - * / ^`、比较、`&& ||`、三元 `a ? b : c`、函数调用、`IndClass.xxx`、`adv20` 这类变量
- 两种方言的函数名映射到同一套算子（`correlation`/`CORR`、`ts_rank`/`TSRANK`……）
- 测试：运算优先级、三元嵌套、两种方言同一公式结果相同

### 任务 3：输入宽表

`qt/panel.py`，测试 `tests/test_panel.py`。

- `load_panel(start, end, universe="main_board") -> dict[str, pd.DataFrame]`：`open, high, low, close, vwap, volume, amount, returns, turn, float_mcap, total_mcap, industry_l1/l2/l3, benchmark_close/open`，以及估值和财务字段
- 测试：PIT（财务在公告日当天不可见）、宽表形状、主板过滤

### 任务 4：Alpha101、Alpha191 登记

`qt/library.py` 的 `ALPHA101`、`ALPHA191`，公式原文存在 `qt/formulas/alpha101.txt`、`qt/formulas/alpha191.txt`。

- 每个公式都能解析；暂缓的（Alpha191 #30）标注原因
- 笔误修正逐条记在登记表的 `note` 字段

### 任务 5：价量补充、估值财务规模因子

`qt/library.py` 的 `PRICE_EXTRA`、`FUNDAMENTAL`，每个因子一个函数，附出处。

### 任务 6：抽样核对

20 个 Alpha101 / Alpha191 因子：在 5 只股票 × 60 天的真实数据上，用独立手写的 pandas 计算逐值对比（相对误差 < 1e-8）。

### 任务 7：全库体检和因子清单

`scripts/build_factor_library.py`：在训练段（2010–2018）全 A 主板上算出全部因子，存到 `data/cache/factors/`（不进仓库），输出 `reports/factor_library/README.md`：

- 每个因子的族、大类、出处、覆盖率、唯一值个数、无穷值比例、计算耗时
- 不输出任何和收益有关的数字

日线数据补齐之前，用现有的 2670 只先跑通，补齐后重跑。
