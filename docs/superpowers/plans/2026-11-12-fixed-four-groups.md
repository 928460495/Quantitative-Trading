# 计划 3d：四组因子等权、参数事先定死 实施计划

**目标**：用重新筛选通过的 16 个因子，按"便宜、跌得多、冷门、业绩超预期"四组等权合成，持股 15 只、每 20 个交易日调仓、一直满仓；所有参数事先按理由定死，数据只用来否决。

**为什么这样做、规则原文**：`docs/DECISIONS.md` 2026-11-12「计划 3d 事先登记的规则」。用户 2026-11-12 把方案决定权交给我，所以这份计划没有等用户逐条确认；用户随时可以改。

**做法**：因子值、行情、行业、市值都已经在 `data/cache/pool/<版本>/{train,valid}/`（计划 3c 存的，59 个因子里包含这 16 个和 vol60），不重算。只补两张标记表（ST、亏损且营收不足 3 亿元），然后用真实账户回测 `qt.account.run_account` 跑主方案和事先列好的单项改动。不用快速模拟器，不挑参数。

**技术栈**：pandas 3、现有的 `qt.composite.neutral_composite`、`qt.account`、`qt.diagnostics`、`qt.trials`。

## 全局约束

- 测试段（2022-01-01 起）不碰；取数都经过 `check_range()`
- 只用训练段（2010–2018）和中途检查段（2019–2021）；中途检查只跑一次
- 单进程；`qt/`、`scripts/` 先写测试，写完故意改错几处确认测试能抓到
- 每个任务跑 `python -m pytest -q -m "not slow and not data"`，提交并推送
- 任何一条检查不满足：写报告、退出码 1、停下来和用户讨论，不改规则
- 单项改动的结果只报告，不换主方案

## 重点复查（测试要盖住的）

1. 排除只作用在该作用的地方：高波动只挡买入，不逼卖；ST、亏损小营收、低价既挡买入也在调仓日卖出
2. 错开起点的版本，基准要按同一段日期算，不能拿全段的基准去比
3. 去掉一组后，合成分确实不含那一组的因子
4. 标记表不能用到未来：财务数据按公告日对齐（沿用 `qt.fundamentals.align_to_days`）
5. 中途检查拒绝重复运行；训练段没通过时不能跑

---

### 任务 1：主板 ST 涨跌幅 2026-07-06 起 10%

**文件**：改 `qt/limits.py`；测试 `tests/test_limits.py`

- [ ] 测试 `test_main_board_st_limit_is_10pct_from_2026_07_06`：`SH600000`、ST，2026-07-03 → 0.05，2026-07-06 → 0.10；非 ST 两天都是 0.10；创业板 ST 两天都是 0.20
- [ ] 看它失败 → `limit_pct` 里加 `ST_LIMIT_REFORM = pd.Timestamp("2026-07-06")`，主板 ST 在这天及以后取 0.10 → 通过
- [ ] 全部测试、提交

### 任务 2：`qt/plan3d.py`——合成分、排除、账户规则、单项改动清单

**文件**：新建 `qt/plan3d.py`；测试 `tests/test_plan3d.py`

接口：

```python
GROUPS: dict[str, tuple[str, ...]]          # 四组，共 16 个因子，名字同 DECISIONS
VOL_FACTOR = "vol60"
REVENUE_FLOOR = 3e8

@dataclass(frozen=True)
class Spec:
    groups: tuple[str, ...] = tuple(GROUPS)
    n_hold: int = 15
    rebalance_every: int = 20
    sell_mult: int = 3
    stop_loss: float | None = None
    vol_cut: float | None = 0.20            # None = 不剔除高波动
    price_floor: float | None = 2.0         # None = 不设
    drop_loss_small: bool = True
    cost_multiplier: float = 1.0
    offset: int = 0                          # 从第几个交易日开始跑（错开调仓起点）

MAIN = Spec()
VARIANTS: dict[str, Spec]                    # 主方案 + DECISIONS 里列的全部单项改动
CHECKS: dict[str, tuple[str, ...]]           # 检查名 → 要高于基准的版本名

def loss_small_revenue(net_profit_ttm, deduct_ttm, revenue_ttm, floor=REVENUE_FLOOR) -> pd.DataFrame  # bool
def high_vol(vol_score, cut) -> pd.DataFrame                                                          # bool
def composite_score(seg, groups) -> pd.DataFrame
def market(seg, spec) -> MarketData
def rules(spec) -> AccountRules
def run(seg, spec) -> AccountResult
def benchmarks(seg, nav, every) -> dict[str, float]   # {"能持有等权": 年化, "沪深300全收益": 年化}，和 nav 同一段日期
def evaluate(seg, variants=VARIANTS, log=None) -> tuple[pd.DataFrame, dict[str, AccountResult]]
def check_table(table) -> pd.DataFrame                 # 每条检查一行，列：最差的版本、它的超额、满足
```

`seg` 是 `build_composite.load_segment` 的结果，另加 `seg["is_st"]`、`seg["loss_small"]` 两张布尔表。

- [ ] 测试（小假数据，在 `fake_seg` 上补 16 个因子、vol60、两张标记表）：
  - `test_groups_cover_16_registered_factors_without_price_level`
  - `test_main_spec_matches_registered_values`（15、20、3、无止损、0.20、2.0）
  - `test_loss_small_needs_both_loss_and_small_revenue`（只亏损、只营收小、数据缺失都不算）
  - `test_high_vol_flags_lowest_fifth_of_vol_score`
  - `test_composite_ignores_dropped_group`（改动被去掉那组的因子值，分数不变）
  - `test_high_vol_blocks_buy_but_keeps_score`、`test_st_and_loss_small_and_cheap_lose_score_and_buy`
  - `test_rules_have_no_stop_loss_no_drawdown_control`
  - `test_offset_starts_later_and_benchmark_uses_same_window`
  - `test_variants_differ_from_main_in_exactly_one_field`、`test_checks_reference_existing_variants`
  - `test_check_table_fails_when_any_member_trails_benchmark`
- [ ] 逐个看失败 → 实现 → 通过；改错检查（至少：去掉 vol 过滤、把逼卖用到高波动上、基准用全段、止损没关、检查里用平均代替"每个都要"）
- [ ] 全部测试、提交

### 任务 3：标记表脚本

**文件**：改 `qt/panel.py`（加 `load_fundamentals(dates, instruments, full_a_dir)`，先 `check_range`）；新建 `scripts/build_plan3d_flags.py`；测试 `tests/test_build_plan3d_flags.py`、`tests/test_panel.py`

- `build_flags(close, is_st_long, fundamentals) -> dict[str, pd.DataFrame]`：按缓存里 close 的日期和股票对齐；ST 缺失按否
- 营收单位核对：全市场滚动四季营收中位数不在 1 亿 ~ 1000 亿元之间就退出（防单位弄错）
- 输出 `data/cache/pool/<版本>/<段>/is_st.parquet`、`loss_small.parquet`，并打印每年被标记的比例
- [ ] 测试：对齐、缺失按否、单位核对会拦住、`load_fundamentals` 碰到测试段报错 → 实现 → 改错检查 → 提交
- [ ] 在主目录运行：`python scripts/build_plan3d_flags.py`

### 任务 4：训练段脚本

**文件**：新建 `scripts/run_plan3d.py`；测试 `tests/test_run_plan3d.py`

`run_train(seg, out_dir, log_path) -> int`：跑 `VARIANTS` 全部版本 → `variants.csv`（年化、两个基准、超额、最大回撤、换手、年成本）→ 四条检查 → 主方案诊断（`qt.diagnostics.diagnose`）和每日净值 `backtest.csv` → 全部满足才写 `final.json`（主方案的参数），否则报告里写"停下讨论"、退出码 1。

- [ ] 测试：正常路径写出 final.json 和报告、试验日志每个版本一条；任何检查不满足时不写 final.json 且退出码 1；旧的 final.json 会被清掉
- [ ] 实现、改错检查、提交
- [ ] 在主目录运行：`python scripts/run_plan3d.py --segment train`，结果提交

### 任务 5：中途检查（2019–2021，只跑一次）

`run_valid(seg, final, out_dir, log_path) -> int`（同一个脚本，`--segment valid`）：没有 final.json 不跑；输出目录非空拒绝运行（先写 STARTED 占位）；跑主方案和 3 个错开版本；4 个版本平均年化 < 能持有等权 − 5 个百分点 → 退出码 1。

- [ ] 测试：拒绝重跑、没有 final.json 报错、明显坏掉时退出码 1 → 实现 → 改错检查 → 提交
- [ ] 训练段四条检查都满足后，在主目录运行：`python scripts/run_plan3d.py --segment valid`，结果提交

### 任务 6：记录和汇报

- [ ] `DECISIONS.md` 追加结果；`HANDOFF.md`、`plans/README.md` 更新
- [ ] 向用户汇报；最终测试等用户批准（`QT_UNLOCK_TEST=1`，只跑一次），这份计划不包含
