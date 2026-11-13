# 稳健、低换手选股 实施计划（计划 3c）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 3b 已通过的 59 个因子，在训练段按滚动前推选出一个池内市值/行业中性、交易少、带回撤分档减仓的组合，固定后在验证段只看一次，按五条标准预演，交给用户审核。

**Architecture:** 新增一个不按手取整的快速模拟器（`qt/fastsim.py`），所有"选"（逐步加因子、216 个组合参数、CSCV）都在它上面跑；`qt/account.run_account` 增加调仓间隔、最小交易额、回撤分档三个规则，只用来确认最终选中的方案和验证段。回撤分档的状态机（`qt/drawdown.py`）两边共用，保证两边规则一样。

**Tech Stack:** Python 3.12、pandas 3、numpy、pyqlib 0.9.7（只在取数时），pytest。conda 环境 `qt`。

**Spec:** `docs/superpowers/specs/2026-11-10-plan3c-design.md`（执行前先读；本计划按它展开）

## Global Constraints

- 测试段（2022-01-01 起）封存：取数一律经过 `qt/config.check_range()` / `segment()`；本计划不解锁
- 训练段 2010-01-04 ~ 2018-12-28，验证段 2019-01-02 ~ 2021-12-31；因子提前一年算（`build_composite.warmup_start`）
- 账户：10 万元；100 股一手；成本 `qt/costs.trade_cost`（佣金万 2.5、最低 5 元，印花税 2023-08-28 前 0.1%，过户费，滑点 0.1%），**不按免五**
- 执行：T 日信号，T+1 收盘成交；`can_buy` / `can_sell` 按信号日对齐（表示 T+1 能不能买卖）
- 个股止损 −20%、冷却 20 天、单股上限 20% 保持 `AccountRules` 默认值
- 原来的回撤兜底 C 不再使用：本计划所有回测 `drawdown_stop=None`
- 均线控仓 A、波动率控制 B 不再跑
- 长任务单进程；`qt/`、`scripts/` 先写测试；跑测试 `python -m pytest -q -m "not slow and not data"`；每个任务结束提交并推送到 `claude/factor-library`
- 任何影响结果的规则运行前写进 `docs/DECISIONS.md`；每个被评估过的方案记试验日志（`qt/trials.log_trial`）
- 提交、代码里不写模型名称

## Review Focus

1. **验证段被跑第二次**：验证段脚本发现 `reports/plan3c/valid/README.md` 已存在就拒绝运行（任务 10 测试）
2. **快速模拟器和真实账户对不上**：同一方案两边差太多，选出来的东西在真实账户上不成立 → 任务 4 用假数据测一致性；任务 8 在真实数据上比参考方案，年化差超过 2 个百分点就停下讨论
3. **回撤分档在历史不足 250 天时**：账户刚开始，最高点只用已有的天数，不报错、不当成回撤 0 以外的值（任务 3 测试）
4. **调仓日碰到涨停 / 跌停**：调仓日买不进的跳过、卖不出的留着，下个调仓日再处理，两个模拟器行为一致（任务 3、4 测试）
5. **逐步加因子时某个因子在窗口前几年全缺**（成长类要 8 期财报）：合成分按"缺的不参与"处理，不让整只股票消失，也不报错（任务 5 测试）

---

### 任务 1：登记规则、更新说明书

**Files:**
- Modify: `docs/DECISIONS.md`（追加一条"计划 3c 事先登记的规则"）
- Modify: `docs/STRATEGY.md` 第 2 节（成功标准改成五条）、第 4 节（股票池：池内市值行业中性）、第 6 节（仓位控制：回撤分档，A/B/C 不再使用）
- Modify: `docs/superpowers/plans/README.md`（3c 行：计划链接、状态"进行中"）、`docs/HANDOFF.md`（下一步指向本计划）

- [ ] **Step 1: 在 DECISIONS 追加下表全文**（数值照抄，跑完不改）

| 项目 | 规则 |
|---|---|
| 候选因子 | `reports/screening/full_a_train_neutral/all_candidates.csv` 的 59 个；方向、大类取 `reports/single_factor/full_a_train_neutral/summary.csv`；中性化同 3b |
| 合成 | 选中因子按大类类内等权 → 每类标准化 → 类间加权平均（低波动权重 w_lv，其余 1）→ 对申万一级行业和对数流通市值中性化 → 平滑：过去 k 天平均后再每天标准化 |
| 前推 | 年份 2013–2018；每年用 2010-01-04 到上一年末的数据做选择，跑下一年；之后用 2010–2018 全部数据按同一程序做最终选择并固定 |
| 比较指标 | 快速模拟器扣成本后的夏普比率（日收益，无风险利率 0，年化 √252） |
| 逐步加因子 | 参考组合：N=20、每 5 天调仓、k=1、跌出前 3N 才卖、行业上限 30%、w_lv=1、回撤分档开启；从空集开始，每步加入使夏普最高的因子；提高不到 0.05 停止 |
| 组合网格（216 个） | N ∈ {15, 20}；调仓间隔 ∈ {1, 5, 10} 个交易日；k ∈ {1, 5, 10}；卖出名次倍数 m ∈ {3, 4, 6}（跌出前 m×N 名才卖）；行业上限 ∈ {0.30, 0.20}；w_lv ∈ {1, 2} |
| 回撤分档 | 账户净值相对过去 250 个信号日（含当天，不足用已有天数）最高点的回撤 d：d < 10% 满仓；10% ≤ d < 15% 上限 floor(N×2/3)；15% ≤ d < 20% 上限 floor(N/3)；d ≥ 20% 上限 0。加深立即生效；回到较浅一档要求 d < 该档下沿 − 2 个百分点。T 日判断、T+1 调整，减仓卖排名最低的 |
| 最小交易额 | 单笔买入预算 < 2,000 元不买 |
| 过拟合 | CSCV：训练段日收益分 16 块，216 个组合；PBO > 0.3 停下讨论，不退回默认 |
| 一致性 | 参考组合在训练段上快速模拟器和 `run_account` 年化相差 > 2 个百分点：停下讨论 |
| 研究关卡 | 前推净值（2013–2018）年化 > 同期能买股票等权（`wide_eval.buyable_benchmark`）年化，且最终合成分和对数流通市值的每日横截面 Spearman 相关取绝对值后的均值 < 0.1；不满足停下讨论 |
| 验证段 | 只跑一次：固定方案 + 回撤分档（另报"不做回撤控制"）；对照 Alpha158 + LightGBM 用同样的 k、调仓间隔、m、N、行业上限和回撤分档；五条标准预演；DSR 试验次数 = 试验日志里不同配置数（含计划 4） |
| 试验日志 kind | `plan3c_step`（每次逐步候选评估）、`plan3c_grid`（每个网格组合在每个前推窗口）、`plan3c_valid` |

- [ ] **Step 2: 改 STRATEGY.md 三节、计划总览、HANDOFF**，内容和设计文档 2、4、5.5 节一致
- [ ] **Step 3: 提交推送**

```bash
git add docs && git commit -m "计划 3c 任务 1：登记规则，更新说明书" && git push
```

### 任务 2：时间对齐和 IC 衰减

**Files:**
- Modify: `qt/wide_eval.py`（`forward_returns` 加 `lag`；新增 `ic_decay`）
- Test: `tests/test_wide_eval.py`

**Interfaces:**
- Produces: `forward_returns(close, horizons, lag: int = 1) -> dict[int, DataFrame]`（T+lag 收盘买、持有 h 天；lag=1 和现在完全一样）；`ic_decay(score: DataFrame, close: DataFrame, can_buy: DataFrame, horizons=(1, 5, 10, 20, 40), lag: int = 1) -> pd.Series`（index=horizon，值=RankIC 均值，只在能买的股票里）

- [ ] **Step 1: 写失败的测试**

```python
def test_forward_returns_lag_default_unchanged_and_no_lookahead():
    close = pd.DataFrame({"a": [1.0, 2, 4, 8, 16, 32]}, index=pd.bdate_range("2020-01-01", periods=6))
    base = forward_returns(close, [2])[2]
    assert base.equals(forward_returns(close, [2], lag=1)[2])
    lag0 = forward_returns(close, [2], lag=0)[2]
    assert lag0.iloc[0, 0] == 3.0                       # 1 → 4
    bumped = close.copy(); bumped.iloc[4:] *= 10
    assert forward_returns(bumped, [2], lag=0)[2].iloc[1, 0] == lag0.iloc[1, 0]   # 只用到第 3 天

def test_ic_decay_perfect_signal():
    # score 等于未来 h 天收益排名时，对应 horizon 的 IC 为 1
    ...  # 构造 3 只股票、30 天，score = forward_returns(close,[5])[5]；断言 ic_decay(...)[5] == pytest.approx(1.0)
```

- [ ] **Step 2: 运行确认失败** `python -m pytest tests/test_wide_eval.py -q` → `TypeError: unexpected keyword 'lag'`
- [ ] **Step 3: 实现**：`forward_returns` 入场价 `close.shift(-lag)`，出场 `close.shift(-(h+lag))`；`lag < 0` 报错；`ic_decay` 对每个 horizon 调 `rank_ic(score, fwd, can_buy).mean()`
- [ ] **Step 4: 运行确认通过**，再跑全部快速测试
- [ ] **Step 5: 提交推送**

### 任务 3：回撤分档、调仓间隔、最小交易额（真实账户）

**Files:**
- Create: `qt/drawdown.py`
- Modify: `qt/account.py`（`AccountRules` 加字段；`run_account` 用它们）
- Test: `tests/test_drawdown.py`、`tests/test_account.py`

**Interfaces:**
- Produces:
  - `qt/drawdown.py`：`DEFAULT_TIERS: tuple[tuple[float, float], ...] = ((0.10, 2/3), (0.15, 1/3), (0.20, 0.0))`；`class DrawdownTiers(tiers=DEFAULT_TIERS, window: int = 250, buffer: float = 0.02)`，方法 `update(nav: float) -> float`（喂入信号日净值，返回持股比例 1 / 2/3 / 1/3 / 0），`@staticmethod slots(n_hold: int, fraction: float) -> int`（`floor(n_hold*fraction + 1e-9)`）
  - `AccountRules` 新字段（默认值保持旧行为）：`rebalance_every: int = 1`、`min_trade: float = 0.0`、`drawdown_tiers: tuple | None = None`、`tier_window: int = 250`、`tier_buffer: float = 0.02`
  - `run_account` 语义：`s % rebalance_every == 0` 的信号日才做"排名卖出"和买入；退市、止损、仓位/分档超额的卖出每天都做；买入预算 < `min_trade` 跳过；`drawdown_tiers` 不为 None 时位置上限 = min(exposure 上限, 分档上限)，此时 `drawdown_stop` 必须为 None（否则 `ValueError`）；分档减仓卖出 reason 记 `"tier"`

- [ ] **Step 1: 写失败的测试**

```python
def test_tiers_deepen_immediately_and_recover_with_buffer():
    t = DrawdownTiers()
    assert t.update(100) == 1.0
    assert t.update(89) == pytest.approx(2/3)      # 回撤 11%
    assert t.update(84) == pytest.approx(1/3)      # 16%
    assert t.update(79) == 0.0                     # 21%
    assert t.update(81.5) == 0.0                   # 18.5%：没低于 20% − 2%，仍空仓
    assert t.update(82.1) == pytest.approx(1/3)    # 17.9% < 18%：回到 1/3 档
    assert t.update(92.1) == 1.0                   # 7.9% < 8%：直接回到满仓（可一次跨多档）
    t2 = DrawdownTiers(); t2.update(100)
    assert t2.update(90) == pytest.approx(2/3)     # 正好 10%：算进 10–15% 档

def test_tiers_peak_window_forgets_old_high():
    t = DrawdownTiers(window=3)
    for v in (100, 90, 90, 90): f = t.update(v)
    assert f == 1.0                                # 100 已出窗口，最高点 90

def test_tiers_short_history_uses_available_days():
    assert DrawdownTiers().update(50) == 1.0

def test_slots_floor():
    assert DrawdownTiers.slots(20, 2/3) == 13 and DrawdownTiers.slots(15, 1/3) == 5

def test_rebalance_every_blocks_rank_sells_and_buys_between_rebalance_days(): ...
    # 5 只股票、N=1、rebalance_every=3：第 2 个信号日排名变了也不换；第 3 个信号日才换
def test_stop_loss_still_fires_on_non_rebalance_day(): ...
def test_min_trade_skips_small_budget(): ...   # 现金 1,500、min_trade=2,000 → 不买
def test_tiers_cut_holdings_next_day_and_sell_lowest_rank(): ...
def test_tiers_with_drawdown_stop_raises():
    with pytest.raises(ValueError):
        run_account(data, AccountRules(drawdown_stop=0.2, drawdown_tiers=DEFAULT_TIERS))
def test_rebalance_day_limit_down_keeps_position_until_next_rebalance(): ...
```

（`...` 处用 `tests/test_account.py` 已有的小行情构造方式写，断言写在注释里的那个行为。）

- [ ] **Step 2: 运行确认失败** → `ModuleNotFoundError: qt.drawdown`
- [ ] **Step 3: 实现 `qt/drawdown.py`**：内部用 `collections.deque(maxlen=window)` 存净值；当前档位只在"加深"或"d < 档位下沿 − buffer"时变化
- [ ] **Step 4: 改 `qt/account.py`**：在"0. 仓位"一段按新字段算 `slot_limit`；"1. 卖"里排名卖出只在调仓日；"2. 买"只在调仓日、预算 < `min_trade` 跳过
- [ ] **Step 5: 运行任务 3 测试和全部快速测试**（旧测试必须全过：默认值保持旧行为）
- [ ] **Step 6: 故意改错两处**（分档阈值用 ≤ 换成 <、调仓日判断用 s 换成 i），确认测试抓到，再改回
- [ ] **Step 7: 提交推送**

### 任务 4：快速模拟器

**Files:**
- Create: `qt/fastsim.py`
- Test: `tests/test_fastsim.py`

**Interfaces:**
- Consumes: `qt/drawdown.DrawdownTiers`；`qt/costs.trade_cost`；`qt/wide_eval._market_data` 产生的 `MarketData`
- Produces: `@dataclass(frozen=True) class SimParams: n_hold: int = 20; rebalance_every: int = 5; sell_mult: int = 3; industry_cap: float = 0.30; tiers: bool = True; min_trade: float = 2000.0; cost_multiplier: float = 1.0`；`simulate(data: MarketData, p: SimParams, cash: float = 100_000) -> pd.Series`（日收益，index=dates，第一天 0）；`sharpe(returns: pd.Series) -> float`

规则和 `run_account` 相同，只有三点简化：不按手取整（买入金额 = 预算）；行业上限按只数算（每个行业最多 `floor(industry_cap*n_hold)` 只）；不记逐笔交易明细。成本仍用 `trade_cost` 按每笔金额算（所以最低 5 元照样生效）。

- [ ] **Step 1: 写失败的测试**

```python
def test_fastsim_matches_account_when_lots_negligible():
    # 30 只股票、200 天随机游走，raw_close 全为 0.01 元（按手取整误差可忽略），cash=1e6
    p = SimParams(n_hold=5, rebalance_every=5, sell_mult=2, industry_cap=1.0, tiers=True, min_trade=0)
    r_fast = simulate(data, p, cash=1e6)
    acct = run_account(data, AccountRules(cash=1e6, n_hold=5, sell_rank=10, max_industry_weight=1.0,
                       rebalance_every=5, drawdown_stop=None, drawdown_tiers=DEFAULT_TIERS))
    nav_fast = 1e6 * (1 + r_fast).cumprod()
    assert (nav_fast / acct.daily["nav"] - 1).abs().max() < 0.005

def test_fastsim_respects_can_buy_and_can_sell(): ...   # 涨停不买、跌停不卖
def test_fastsim_min_commission_applies(): ...           # 1 笔 5,000 元买入当天成本 = trade_cost(5000,"buy",d)
def test_fastsim_industry_cap_by_count(): ...            # 同一行业 10 只最高分，cap 0.2、N=10 → 只买 2 只
def test_fastsim_runs_fast():                           # 2,200 天 × 3,000 只 < 3 秒（标 slow）
```

- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现**：按日循环，持仓用"列号 → 持仓市值"字典，排名用 `qt.account._ranks`；每天先按收益更新市值，再按信号日处理卖、买；分档用 `DrawdownTiers`
- [ ] **Step 4: 运行确认通过**；速度测试单独跑 `python -m pytest tests/test_fastsim.py -q -m slow`
- [ ] **Step 5: 故意改错两处**（成本用固定比例、忽略 can_sell），确认一致性测试或专项测试抓到
- [ ] **Step 6: 提交推送**

### 任务 5：中性化合成、平滑、低波动加权

**Files:**
- Modify: `qt/composite.py`
- Test: `tests/test_composite.py`

**Interfaces:**
- Consumes: `qt/wide_eval.neutralize`、`zscore`
- Produces: `composite(cat_scores, weights: Mapping[str, float] | None = None) -> DataFrame`（不传 weights 和现在完全一样）；`neutral_composite(factor_scores: Mapping[str, DataFrame], categories: Mapping[str, str], industry: DataFrame, log_mcap: DataFrame, lowvol_weight: float = 1.0, smooth: int = 1) -> DataFrame`（类内等权 → 每类标准化 → 类间加权，"低波动"类权重 = lowvol_weight → `neutralize(..., size=True)` → 过去 `smooth` 天平均（只用当天及以前，不足 smooth 天用已有天数）→ 每天 `zscore`）

- [ ] **Step 1: 写失败的测试**

```python
def test_composite_weights_default_unchanged(): ...       # weights=None 与现有结果逐值相等
def test_lowvol_weight_two_counts_double():
    # 两类：低波动=A、价值=B，w=2 → 中性化前 = zscore((2A+B)/3)
def test_neutral_composite_uncorrelated_with_size():
    # 合成分构造成 = log_mcap + 噪声；结果每天和 log_mcap 的相关 < 1e-8
def test_smooth_uses_only_past(): ...                      # 改动第 t+1 天的因子值，第 t 天结果不变
def test_factor_missing_early_years_does_not_drop_stock(): ...  # 某因子前 100 天全缺，股票当天仍有分数（Review Focus 5）
```

- [ ] **Step 2–4: 失败 → 实现 → 通过**
- [ ] **Step 5: 提交推送**

### 任务 6：59 个因子的数据和"买不买得起"检查

**Files:**
- Create: `scripts/build_pool.py`
- Modify: `scripts/build_composite.py`（`build_segment` 加参数 `save_log_mcap: bool = False`，为 True 时存 `log_mcap.parquet`；`load_segment` 有就读）
- Test: `tests/test_build_pool.py`

**Interfaces:**
- Consumes: `build_composite.build_segment`、`load_segment`、`run_full_a_screen.load_context`
- Produces: `read_pool(candidates: Path, summary: Path) -> DataFrame`（index=因子，列 category、direction；59 行）；`lot_feasibility(raw_close: DataFrame, can_buy: DataFrame, nav: float, n_hold: int) -> pd.Series`（每年"能买的股票里，一手金额超过 nav/n_hold 的比例"）；缓存目录 `data/cache/pool/<数据版本>/<段>/`，结构同 composite 缓存，多一个 `log_mcap.parquet`，不写 `composite.parquet`
- 命令：`python scripts/build_pool.py [--segment train|valid|both]`，结束打印 N=15、20 时的 `lot_feasibility`

- [ ] **Step 1: 测试**：`read_pool` 读出 59 行且方向只取 ±1；用假的 `compute` 跑 `build_segment(save_log_mcap=True)` 后 `load_segment` 读回 `log_mcap`；`lot_feasibility` 在价格 60 元、nav 100,000、N=20 时比例为 1.0
- [ ] **Step 2–4: 失败 → 实现 → 通过**
- [ ] **Step 5: 提交推送**
- [ ] **Step 6: 用户本机运行**（单进程，预计 1–2 小时）：

```bash
python scripts/build_pool.py
```

把打印的"买不起比例"贴回来；N=20 时某年超过 20% 就停下讨论（会让持股达不到 N）。

### 任务 7：CSCV 回测过拟合概率

**Files:**
- Create: `qt/pbo.py`
- Test: `tests/test_pbo.py`

**Interfaces:**
- Produces: `cscv_pbo(returns: pd.DataFrame, n_blocks: int = 16) -> dict`（columns=方案，index=日期；按时间切 n_blocks 块，取所有 C(n_blocks, n_blocks/2) 种一半作样本内、另一半样本外；每种组合找样本内夏普最高的方案，算它在样本外的相对名次 ω，logit λ=ln(ω/(1−ω))；返回 `{"pbo": λ≤0 的比例, "n_combos": int, "logits": np.ndarray}`）；`n_blocks` 必须是偶数

- [ ] **Step 1: 测试**：一个方案每天都比其他方案高 0.1% 且独立 → pbo < 0.05；200 个纯噪声方案 → 0.3 < pbo < 0.7；`n_blocks=16` 时 `n_combos == 12870`；奇数 `n_blocks` 报错
- [ ] **Step 2–4: 失败 → 实现 → 通过**（样本外名次 ω 用 (rank)/(N+1)）
- [ ] **Step 5: 提交推送**

### 任务 8：诊断和标准

**Files:**
- Create: `qt/diagnostics.py`
- Test: `tests/test_diagnostics.py`

**Interfaces:**
- Produces:
  - `rolling_shortfall(nav: pd.Series, bench: pd.Series, months: int = 36) -> pd.DataFrame`（按月末滑动，每个完整 36 个月窗口一行：start、end、策略年化、基准年化、shortfall = 基准年化 − 策略年化）
  - `criteria(nav, bench, nav_cost2x, baseline_annual: float) -> pd.DataFrame`（五行：①–⑤，列"数值""门槛""满足"；④ 取 `rolling_shortfall` 最大值和 0.05 比）
  - `diagnose(result: AccountResult | pd.Series, seg: dict, score: DataFrame) -> dict[str, pd.DataFrame]`（扣成本前后、相对 `buyable_benchmark` 的分年超额、分年收益和回撤、`rolling_shortfall`、合成分和 `log_mcap` 的日相关均值、交易笔数、换手、成本、分档触发天数）
  - `write_report(sections: dict[str, pd.DataFrame], path: Path, title: str, notes: list[str]) -> None`

- [ ] **Step 1: 测试**：36 个月正好一个窗口；策略和基准相同 → shortfall 全 0；基准每年多 6% → ④ 不满足；`criteria` 五行齐全
- [ ] **Step 2–4: 失败 → 实现 → 通过**
- [ ] **Step 5: 提交推送**

### 任务 9：训练段前推、网格、CSCV、研究关卡

**Files:**
- Create: `qt/walkforward.py`、`scripts/run_plan3c_train.py`
- Test: `tests/test_walkforward.py`、`tests/test_run_plan3c_train.py`

**Interfaces:**
- Consumes: 任务 4 `simulate`、`SimParams`、`sharpe`；任务 5 `neutral_composite`；任务 6 缓存；任务 7 `cscv_pbo`；任务 8 `diagnose`、`write_report`；`qt/trials.log_trial`
- Produces:
  - `REFERENCE = SimParams(n_hold=20, rebalance_every=5, sell_mult=3, industry_cap=0.30)`；`GRID: list[tuple[SimParams, int, float]]`（216 个 (参数, smooth, lowvol_weight)，取值见任务 1）
  - `forward_select(seg, pool, window: tuple[str, str], min_gain: float = 0.05, log=None) -> list[str]`
  - `grid_select(seg, factors: list[str], window) -> tuple[pd.DataFrame, tuple]`（216 列日收益，最好的那个配置）
  - `walk_forward(seg, pool, years=range(2013, 2019)) -> dict`（每年选中的因子和配置、拼接的 2013–2018 前推日收益；每年结果存 `data/cache/plan3c/wf_<year>.json`，已存在就跳过，可断点续跑）
  - 命令：`python scripts/run_plan3c_train.py`；输出 `reports/plan3c/train/README.md`、`walk_forward.csv`、`grid.csv`、`ic_decay.csv`，以及只在所有检查都通过时写的 `final.json`（`{"factors": [...], "params": SimParams 字段, "smooth": int, "lowvol_weight": float}`，任务 10 读它）
  - 说明：设计文档 5.4 的"每次最多换掉几只"不进网格，本计划不实现（买入门槛严、持有门槛松已经限制了换手）

脚本顺序：①参考组合快速模拟 vs `run_account` 年化差（> 2 个百分点停）②各类 `ic_decay`（只报告，说明"按速度配"由每步在参考组合下比夏普实现）③`walk_forward` ④用 2010–2018 全部数据跑 `forward_select` + `grid_select` 得最终方案 ⑤最终因子下 216 个组合全训练段日收益做 `cscv_pbo`（> 0.3 停）⑥研究关卡（不满足停）⑦最终方案 `run_account` 跑训练段 + `diagnose`；壳价值对比（去掉每天对数流通市值最小 30% 的股票后重跑最终方案，只报告）⑧写报告、记日志。"停"= 写报告说明原因并以非零退出码结束，不继续后面的步骤。

- [ ] **Step 1: 测试**（小假数据：3 个因子其中 1 个是真信号、2 个噪声，6 年）：`forward_select` 第一步选真信号、噪声加入提高 < 0.05 时停止；`walk_forward` 每年只用窗口内数据（改动窗口之后的价格，该年选择不变）；断点续跑：第二次运行读缓存不重算；`GRID` 长度 216 且无重复；停的条件触发时退出码非零
- [ ] **Step 2–4: 失败 → 实现 → 通过**
- [ ] **Step 5: 提交推送**
- [ ] **Step 6: 用户本机运行**（单进程；逐步加因子约 3–4 小时，可断点续跑）：

```bash
python scripts/run_plan3c_train.py
```

- [ ] **Step 7: 结果提交推送**；在 DECISIONS 记下前推结果、PBO、研究关卡是否通过、最终固定的因子和配置。任何一个"停"触发：停下和用户讨论，不进任务 10

### 任务 10：验证段（只跑一次）

**Files:**
- Create: `scripts/run_plan3c_valid.py`
- Test: `tests/test_run_plan3c_valid.py`

**Interfaces:**
- Consumes: 任务 9 写在 `reports/plan3c/train/final.json` 的固定方案（因子、SimParams、smooth、lowvol_weight）；任务 8 `criteria`、`diagnose`；`data/cache/baseline/<版本>/score_valid.parquet`；`qt/trials.deflated_sharpe`、`count_unique`
- Produces: `reports/plan3c/valid/README.md`、`backtest.csv`；命令 `python scripts/run_plan3c_valid.py`

内容：固定方案 `run_account`（回撤分档开）；同方案不做回撤控制（只报告）；成本翻倍；对照基准打分按同样的 smooth 平滑后用同样的账户规则；`criteria` 五条；DSR（试验次数 = `count_unique()`，含计划 4）；报告写明"验证段在计划 4 已看过两次"。

- [ ] **Step 1: 测试**：`README.md` 已存在时脚本拒绝运行（退出码非零、不写任何文件、不记日志）；`final.json` 缺失时报错；假数据上五条标准齐全
- [ ] **Step 2–4: 失败 → 实现 → 通过**
- [ ] **Step 5: 提交推送**
- [ ] **Step 6: 用户本机运行一次**：

```bash
python scripts/run_plan3c_valid.py
```

- [ ] **Step 7: 结果提交推送**，DECISIONS 记结果

### 任务 11：审核节点

结果交给用户：五条标准、研究关卡、PBO、DSR、和计划 4 的对照。用户决定是否进入最终测试；最终测试另写计划，只跑一次。
