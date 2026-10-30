# 按真实账户记账的回测 实施计划

> **执行方式：** 按 `CLAUDE.md`，用 superpowers:executing-plans 在当前会话里逐个任务执行。步骤用复选框（`- [ ]`）跟踪。

**目标：** 一个按 10 万元真实账户记账的日频回测引擎，实现说明书里的全部交易规则：持仓只数与缓冲、平分现金、1 手取整、最低佣金、单股和行业上限、个股止损和冷却期、均线仓位控制、账户回撤兜底。

**架构：** 自己写一个纯 pandas / numpy 的账户模拟器 `qt/account.py`，不扩展 Qlib 的回测：Qlib 的交易所只支持统一的涨跌停阈值，我们的真实涨跌停、ST、上限、止损、仓位控制都要绕开它的框架实现，自己写更直接、更快，也更好测。Qlib 回测留作核对：在沪深300 的旧设定下，两者结果要对得上。

**技术：** pandas、numpy；输入是长格式的 `(datetime, instrument)` 序列，内部转成"日期 × 股票"的矩阵逐日循环。

**说明书：** [`docs/STRATEGY.md`](../../STRATEGY.md) 第 5、6、8、9 节。

## 全局约束

- 时间约定：T 日收盘出信号，T+1 收盘成交。第 i 天成交时，所有决定只能用第 i-1 天及以前的数据
- 账户 100,000 元；持仓 15 只、排名掉出前 45 才卖；单股 ≤ 20%、行业 ≤ 30%（申万一级），只在买入时检查
- 买入：当天空出的位置平分可用现金；每只的分配额不超过账户净值的 20%；分到的钱不够 1 手就跳过、换下一名
- 1 手 = 100 股；科创板（SH688、SH689）最少 200 股，之后可以按 1 股加
- 个股止损：信号日复权收盘价比持仓成本跌 20%，下一交易日卖出；卖出后 20 个交易日内不买回
- 成本：佣金 0.025%、每笔最低 5 元；滑点 0.1%（买卖都收）；印花税卖出收，2023-08-28 前 0.1%、之后 0.05%；`cost_multiplier` 整体放大（最低佣金一起放大）
- 记账用复权价：持仓记"复权份额"，分红送转自动体现在复权价里（相当于分红再投资，不计红利税）；手数用未复权价算
- `score` 为 NaN 的股票视为不在股票池（比如 ST）：持有的按"掉出排名"卖出
- 仓位控制：`exposure` 是每个信号日的持股比例（1、0.5、0），持股只数上限 = `floor(15 × exposure)`；账户从最高点回撤 20% 时全部卖出，等均线信号先变"空"再变"满"才回来，最高点重置为当时的净值

## 重点检查

1. **用到未来数据**：改动第 i 天的分数或价格，第 i 天的成交不能变。→ 任务 3 的测试
2. **现金变成负数**：最低佣金让小额买单的成本比例变高，买完现金必须 ≥ 0。→ 任务 3 的测试
3. **持有的股票长期停牌或退市**：停牌期间按最后价格估值，不能卖；数据彻底结束（退市）后按最后价格卖出、原因记"退市"，并在结果里统计次数（偏乐观，报告里要写明）。→ 任务 6 的测试
4. **所有候选都买不起**（1 手太贵）：现金留着，不报错。→ 任务 3 的测试
5. **仓位从满降到空，但有的持仓跌停卖不掉**：卖不掉的留着，下一天再卖；位置数不会变成负数。→ 任务 5 的测试

---

### 任务 1：交易成本

**文件：**
- 修改：`qt/costs.py`
- 测试：`tests/test_costs.py`（新建）

**接口：**
- 产出：
  - `MIN_COMMISSION = 5.0`
  - `trade_cost(value: float, side: str, date: pd.Timestamp, multiplier: float = 1.0) -> float`：`side` 为 `"buy"` 或 `"sell"`；佣金 `max(value × 0.025%, 5)` + 滑点 `value × 0.1%` + 卖出时印花税，整体乘 `multiplier`

- [ ] **步骤 1：写失败的测试**

```python
def test_min_commission_applies_to_small_trades():
    assert trade_cost(6700, "buy", pd.Timestamp("2019-01-02")) == pytest.approx(5 + 6.7)
def test_large_trade_uses_rate():
    assert trade_cost(1_000_000, "buy", pd.Timestamp("2019-01-02")) == pytest.approx(250 + 1000)
def test_sell_adds_stamp_duty_by_date():
    assert trade_cost(10_000, "sell", pd.Timestamp("2023-08-25")) == pytest.approx(5 + 10 + 10)
    assert trade_cost(10_000, "sell", pd.Timestamp("2023-08-28")) == pytest.approx(5 + 10 + 5)
def test_multiplier_scales_everything():
    assert trade_cost(6700, "buy", pd.Timestamp("2019-01-02"), multiplier=2) == pytest.approx(2 * (5 + 6.7))
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 2：输入数据和手数

**文件：**
- 新建：`qt/account.py`
- 测试：`tests/test_account.py`

**接口：**
- 产出：
  - `@dataclass(frozen=True) class AccountRules`：`cash: float = 100_000`、`n_hold: int = 15`、`sell_rank: int = 45`、`max_stock_weight: float = 0.20`、`max_industry_weight: float = 0.30`、`stop_loss: float | None = 0.20`、`cooldown_days: int = 20`、`drawdown_stop: float | None = 0.20`、`cost_multiplier: float = 1.0`
  - `@dataclass class MarketData`：`dates: pd.DatetimeIndex`、`instruments: pd.Index`，以及"日期 × 股票"的 numpy 矩阵 `score`、`adj_close`、`raw_close`（float）、`can_buy`、`can_sell`（bool）、`industry`（object，可以为 None）
  - `market_data(score, adj_close, raw_close, can_buy, can_sell, industry=None) -> MarketData`：参数都是 `(datetime, instrument)` 索引的 Series；`can_buy`、`can_sell` 缺失按 False；价格不填充（停牌为 NaN）
  - `lot_shares(budget: float, raw_price: float, instrument: str) -> int`：普通股票按 100 股取整；科创板不足 200 股为 0，否则按 1 股取整；价格缺失或预算不够为 0

- [ ] **步骤 1：写失败的测试**

```python
def test_lot_shares():
    assert lot_shares(6700, 12.3, "SH600000") == 500
    assert lot_shares(6700, 70.0, "SH600000") == 0
    assert lot_shares(6700, 30.0, "SH688001") == 223
    assert lot_shares(5000, 30.0, "SH688001") == 0      # 不够 200 股
    assert lot_shares(6700, float("nan"), "SH600000") == 0
def test_market_data_shapes_and_missing_masks():
    # 2 天 × 3 只；can_buy 只给了 2 行：其余为 False；adj_close 缺的保持 NaN
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 3：核心循环：缓冲规则、平分现金、上限、成本、记账

**文件：**
- 修改：`qt/account.py`
- 测试：`tests/test_account.py`

**接口：**
- 消费：任务 1 的 `trade_cost`，任务 2 的 `MarketData`、`AccountRules`、`lot_shares`
- 产出：
  - `@dataclass class AccountResult`：`daily: pd.DataFrame`（索引为成交日，列 `nav, cash, stock_value, n_holdings, slot_limit, cost, n_buys, n_sells`）、`trades: pd.DataFrame`（列 `date, instrument, side, shares, price, value, cost, reason`；`reason` 取 `"buy"`、`"rank"`、`"stop"`、`"exposure"`、`"drawdown"`、`"delisted"`）
  - `run_account(data: MarketData, rules: AccountRules = AccountRules(), exposure: pd.Series | None = None) -> AccountResult`
- 每个成交日 i（信号日 s = i-1）的顺序：
  1. 卖：持有的股票，在信号日 s 没有分数或排名 > `sell_rank`，且 `can_sell[s]` → 按第 i 天收盘卖出
  2. 买：空位数 = 位置上限 − 卖完后的持仓数；分配额 = `min(现金 / 空位数, 净值 × max_stock_weight)`；按 s 日排名从高到低，跳过已持有、`can_buy[s]` 为 False、冷却期内、买了行业会超 30% 的，按 `lot_shares` 算股数，0 股就跳过；成本加上后超过现金就少买 1 手
  3. 记账：净值 = 现金 + Σ 复权份额 × 第 i 天复权收盘价（停牌用最后一个有效价）

- [ ] **步骤 1：写失败的测试**（每个场景 2–5 只股票、几天的手工数据）

```python
def test_buys_top_ranked_with_equal_cash_and_lots():
    # 首日 3 个空位、现金 100000：每只分配 33333，按 10 元一手买 3300 股；剩余现金 = 100000 − 买入金额 − 成本
def test_sells_only_when_rank_drops_below_sell_rank():
    # n_hold 2、sell_rank 3：排名掉到第 3 不卖，掉到第 4 卖，原因 "rank"
def test_missing_score_means_out_of_pool_and_sold():
def test_cannot_sell_keeps_position():
def test_stock_cap_limits_allocation():
    # 只有 1 个空位、现金 100000：分配额封顶 20000
def test_industry_cap_skips_to_next():
    # 已有同行业 28%，下一只同行业会超 30%：跳过，买下一只别的行业
def test_all_candidates_too_expensive_keeps_cash():
def test_cash_never_negative_with_min_commission():
    # 现金正好够买 1 手但不够付佣金：少买 1 手（这里变成 0），现金 ≥ 0
def test_no_lookahead():
    # 把最后一天的分数和价格改掉：最后一天的成交记录完全不变（只用前一天的信号）
def test_nav_uses_adjusted_price_for_dividends():
    # 除息日未复权价下跌、复权价不变：净值不变
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 4：个股止损和冷却期

**文件：**
- 修改：`qt/account.py`
- 测试：`tests/test_account.py`

**接口：**
- 持仓成本按复权价记（加权平均）；信号日 s 复权收盘 ≤ 成本 × (1 − `stop_loss`)，且 `can_sell[s]` → 第 i 天卖出，原因 `"stop"`；卖出后 `cooldown_days` 个交易日内不买回；`stop_loss=None` 关掉

- [ ] **步骤 1：写失败的测试**

```python
def test_stop_loss_sells_next_day():
    # 买入成本 10，信号日收盘 7.99（跌 20.1%）：下一天卖出，原因 "stop"；7.99 → 8.01（跌 19.9%）不卖
def test_stopped_stock_not_rebought_within_cooldown():
    # 止损后它排名第一：20 个交易日内不买，第 21 个交易日可以买
def test_stop_loss_blocked_by_limit_down_retries():
def test_stop_loss_disabled():
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 5：仓位控制

**文件：**
- 修改：`qt/account.py`
- 测试：`tests/test_account.py`

**接口：**
- 产出：
  - `trend_exposure(index_close: pd.Series, window: int, low: float) -> pd.Series`：收盘价 ≥ `window` 日均线为 1.0，否则为 `low`；均线还算不出的日子为 1.0
  - `run_account` 的 `exposure`：信号日 → 持股比例；位置上限 = `floor(n_hold × exposure)`；持仓多于上限时按排名从低到高卖掉多出来的，原因 `"exposure"`
  - 账户回撤兜底：信号日净值 ≤ 历史最高 × (1 − `drawdown_stop`) → 第二天起全部卖出（原因 `"drawdown"`），位置上限为 0；直到 `exposure` 先出现 < 1 再出现 = 1，才恢复，最高点重置为恢复那天的净值；`drawdown_stop=None` 关掉

- [ ] **步骤 1：写失败的测试**

```python
def test_trend_exposure():
    # 5 日均线：价格在均线上为 1，下为 low；前 4 天为 1
def test_exposure_half_sells_lowest_ranked():
    # n_hold 4、exposure 从 1 变 0.5：卖掉排名最低的 2 只，原因 "exposure"
def test_exposure_zero_with_limit_down_holding():
    # 降到 0 时 1 只跌停卖不掉：留着，第二天卖；位置上限不为负
def test_drawdown_stop_liquidates_and_waits_for_fresh_signal():
    # 净值跌 20%：第二天全卖；exposure 一直为 1 时不回来；变 0 再变 1 才回来，最高点重置
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 6：退市处理和指标

**文件：**
- 修改：`qt/account.py`
- 测试：`tests/test_account.py`

**接口：**
- 退市：持有股票在数据里最后一个有价格的日子之后（之后再也没有价格），下一个成交日按最后价格卖出，原因 `"delisted"`
- 产出：`account_metrics(result: AccountResult, bench_close: pd.Series) -> dict[str, float]`：`annual_return`（复合年化）、`bench_annual`、`excess_annual`（两者之差）、`max_drawdown`（净值从最高点的最大跌幅）、`volatility`、`sharpe`（无风险利率按 0）、`turnover`（日均卖出金额 / 净值）、`cost_annual`（年化成本 / 净值）、`n_trades`、`n_stops`、`n_delisted`、`avg_exposure`（股票市值 / 净值的平均）

- [ ] **步骤 1：写失败的测试**

```python
def test_delisted_holding_sold_at_last_price():
def test_metrics_on_known_path():
    # 净值 100 → 110 → 99 → 121：max_drawdown = -10%；annual_return 按 252 天复合
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 7：和 Qlib 回测核对

**文件：**
- 新建：`scripts/check_account.py`、`reports/account_check.md`（生成）
- 测试：`tests/test_account.py` 里加一个 `@pytest.mark.data` 的测试

**接口：**
- 消费：第 5 个 PR 的缓存（`data/cache/scores_valid_*.pkl`）和 `qt/backtest.py` 的 `run_backtest`
- 用沪深300 验证段、等权合成分数、旧设定（账户 1 亿、30 只、缓冲 30/90、不设上限、不止损、不控仓、`can_buy` 用 `tradable_mask`），分别跑 `run_account` 和 Qlib 的 `run_backtest`

- [ ] **步骤 1：** 写 `@pytest.mark.data` 的测试：两者扣成本后的年化收益相差 < 1 个百分点，日收益相关系数 > 0.95
- [ ] **步骤 2：** 测试失败（脚本不存在）
- [ ] **步骤 3：** 实现 `scripts/check_account.py`，写 `reports/account_check.md`：两者的年化收益、成本、换手、日收益相关系数；以及差异的原因（Qlib 留 1% 现金、按价格漂移等）
- [ ] **步骤 4：** 测试通过
- [ ] **步骤 5：** 提交，推送，开 PR
