# 全 A 数据层 实施计划

> **执行方式：** 按 `CLAUDE.md`，用 superpowers:executing-plans 在当前会话里逐个任务执行。步骤用复选框（`- [ ]`）跟踪。

**目标：** 补齐全 A 研究需要的数据：ST 标记、交易状态、真实涨跌停、估值、换手率、按发布日期对齐的季度财务，并提供全 A 的股票池过滤。

**架构：** 采集脚本在用户电脑上跑（云端连不上 baostock 和东方财富），原始结果缓存在本地，最后打包成一个 tar 文件，用户上传到本仓库的 GitHub Release；云端脚本下载解包到 `QT_FULL_A_DIR`，`qt/` 里的读取函数按 `(datetime, instrument)` 对齐到 Qlib 数据。数据文件不进 git。

**技术：** baostock（日线）、东方财富数据中心接口（季度财务，直接用 requests 调，不经过 AkShare 的列名改写）、pandas、pyarrow（parquet）。

**说明书：** [`docs/STRATEGY.md`](../../STRATEGY.md) 第 4、9、10 节。

## 全局约束

- 采集端只依赖 `requirements-collect.txt`（numpy、pandas、akshare、baostock、pyarrow、requests），不需要 Qlib
- 股票代码在 `qt/` 里一律用 Qlib 的写法（`SH600000`），转换只经过 `to_instrument()`
- 包含退市股票；采集结果要报告每年的覆盖率，不能悄悄丢
- 财务数据按发布日期对齐：发布日当天收盘后才算知道，**下一个交易日**起可用
- 读取函数都经过 `check_range()`，测试段未解锁时拿不到 2022 年以后的数据（采集和打包本身不受限）
- 写文件先写临时文件再改名，中断不会留下半个文件
- 不关闭 TLS 校验；不在聊天里传密码或令牌
- 日期范围：日线 2009-01-01 起（给 250 日窗口留出预热），财务报告期 2008-03-31 起

## 重点检查

1. **采集中途断网或被限流**：重跑时跳过已完成的股票和报告期，只补缺的；每只股票、每个报告期的失败都记进 `meta.json`。→ 任务 1、2 的续跑测试
2. **退市股缺数据**：东方财富的季报接口可能不含已退市公司。→ 任务 3 的覆盖率表：按年统计"有财务数据的股票 / 当年上市的股票"
3. **同一份报告多次公告（更正、补充）**：每个（股票，报告期）只留最早的公告日期。→ 任务 6 的测试
4. **除权除息日的涨跌停**：涨跌停价按交易所的前收盘价（除权参考价）算，不按前一天的收盘价。→ 任务 5 的测试用除息日例子
5. **代码写法不一致**（`sh.600000`、`600000`、`SH600000`、`bj.`）。→ 任务 1 的 `to_instrument()` 测试，北交所在采集时就排除

---

### 任务 1：baostock 日线采集

**文件：**
- 新建：`scripts/collect_full_a.py`
- 修改：`requirements-collect.txt`（加 baostock、pyarrow、requests）
- 测试：`tests/test_collect_full_a.py`

**接口：**
- 产出：
  - `to_instrument(code: str) -> str`：`"sh.600000"`、`"600000"`、`"SH600000"` → `"SH600000"`；6 位数字按首位推交易所（6、9 → SH；0、2、3 → SZ；4、8 → BJ）
  - `stock_list(bs) -> pd.DataFrame`：列 `instrument, name, ipo_date, out_date`；只要 `type == "1"`（股票），排除北交所
  - `DAILY_FIELDS = "date,code,preclose,close,volume,amount,turn,tradestatus,isST,peTTM,pbMRQ,psTTM,pcfNcfTTM"`，`adjustflag="3"`（不复权）
  - `collect_daily(bs, instruments: list[str], out_dir: Path, start: str = "2009-01-01", end: str | None = None, retries: int = 3, pause: float = 0.0) -> dict[str, str]`：每只股票一个文件 `out_dir / f"{instrument}.csv"`；已存在就跳过；返回 `{instrument: "ok" | "empty" | "failed: <原因>"}`

- [ ] **步骤 1：写失败的测试**（假的 `bs` 模块，仿照 `tests/test_collect.py` 的做法）

```python
def test_to_instrument():
    assert to_instrument("sh.600000") == "SH600000"
    assert to_instrument("sz.000001") == "SZ000001"
    assert to_instrument("300750") == "SZ300750"
    assert to_instrument("688001") == "SH688001"
    assert to_instrument("SH600000") == "SH600000"
    assert to_instrument("bj.430047") == "BJ430047"

def test_stock_list_keeps_only_stocks_and_drops_bj(fake_bs): ...
    # 假数据含 type 1/2、sh/sz/bj；结果只有 sh、sz 的 type 1，列名如上

def test_collect_daily_resumes(tmp_path, fake_bs):
    # 第一次：A 成功、B 抛异常（重试 3 次都失败）；第二次只请求 B
    assert first == {"SH600000": "ok", "SZ000001": "failed: ..."}  # 前缀匹配
    assert fake_bs.calls_second_run == ["sz.000001"]

def test_collect_daily_no_partial_file(tmp_path, fake_bs_crashes_midway):
    # 写到一半异常：目录里没有 .csv，只可能有被清理的临时文件
```

- [ ] **步骤 2：** `pytest tests/test_collect_full_a.py -q`，预期失败（函数不存在）
- [ ] **步骤 3：** 实现上面三个函数。登录失效（返回码非 0）时重新 `bs.login()` 再重试；每只股票之间停 `pause` 秒
- [ ] **步骤 4：** 测试通过
- [ ] **步骤 5：** 提交

### 任务 2：东方财富季度财务采集

**文件：**
- 修改：`scripts/collect_full_a.py`
- 测试：`tests/test_collect_full_a.py`

**接口：**
- 产出：
  - `REPORTS = {"income": ("RPT_DMSK_FN_INCOME", "REPORT_DATE"), "balance": ("RPT_DMSK_FN_BALANCE", "REPORT_DATE"), "cashflow": ("RPT_DMSK_FN_CASHFLOW", "REPORT_DATE"), "yjbb": ("RPT_LICO_FN_CPD", "REPORTDATE")}`（报表名，日期字段名）
  - `report_dates(start: str = "2008-03-31", end: str | None = None) -> list[str]`：季末日期 `"YYYY-MM-DD"`，到 `end` 为止
  - `collect_reports(session, names: list[str], dates: list[str], out_dir: Path, retries: int = 3, pause: float = 0.5) -> dict[str, str]`：调 `https://datacenter-web.eastmoney.com/api/data/v1/get`，`columns=ALL`、`pageSize=500`，翻完所有页；每个（报表，报告期）一个文件 `out_dir / name / f"{date}.csv"`，保留接口返回的全部原始列；已存在就跳过；返回 `{f"{name}/{date}": "ok" | "empty" | "failed: ..."}`
  - 三张 DMSK 表的筛选条件沿用 AkShare：`(SECURITY_TYPE_CODE in ("058001001","058001008"))(TRADE_MARKET_CODE!="069001017")`，再加报告期
  - `INDEXES = ("H00300", "000985")`（沪深300 全收益、中证全指）
  - `collect_indexes(ak, out_dir: Path, start: str = "20090101", end: str | None = None) -> dict[str, str]`：用 `ak.stock_zh_index_hist_csindex` 取每个指数的日收盘价，存 `out_dir / "index" / f"{symbol}.csv"`；返回每个指数的起止日期或失败原因。全收益指数历史不够长时照实记录，不报错（说明书第 2 节：拿不到就用普通指数并写明偏差）

- [ ] **步骤 1：写失败的测试**（假的 `session.get`，返回两页 JSON）

```python
def test_report_dates():
    assert report_dates("2008-03-31", "2008-12-31") == ["2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31"]

def test_collect_reports_reads_all_pages(tmp_path, fake_session):
    status = collect_reports(fake_session, ["income"], ["2015-12-31"], tmp_path)
    df = pd.read_csv(tmp_path / "income" / "2015-12-31.csv")
    assert len(df) == 3 and "NOTICE_DATE" in df.columns   # 两页合计 3 行，原始列名保留
    assert status == {"income/2015-12-31": "ok"}

def test_collect_reports_empty_period(tmp_path, fake_session_empty):
    # 接口返回 result 为 null：记为 "empty"，不写文件，不重试

def test_collect_indexes_records_short_history(tmp_path, fake_ak):
    # 假的 H00300 只从 2013 年开始：文件照写，返回值里记下实际起始日期
```

- [ ] **步骤 2：** 测试失败
- [ ] **步骤 3：** 实现
- [ ] **步骤 4：** 测试通过
- [ ] **步骤 5：** 提交

### 任务 3：打包与覆盖率报告

**文件：**
- 修改：`scripts/collect_full_a.py`（命令行入口：`--part {list,daily,reports,pack,all}`、`--pause`、`--retries`）
- 测试：`tests/test_collect_full_a.py`

**接口：**
- 产出：
  - `pack(raw_dir: Path, out_dir: Path, collected_on: str) -> Path`：生成 `out_dir / f"full_a_{collected_on}.tar"`，里面是
    - `stocks.parquet`（`stock_list` 的结果）
    - `daily.parquet`：全部股票的日线合在一起，`instrument` 列用 Qlib 写法，`date` 为日期，数值列 float32，`tradestatus`、`isST` 为 int8
    - `reports/<name>.parquet`：每张表所有报告期合在一起，原始列
    - `index.parquet`：列 `symbol, date, close`
    - `meta.json`：采集日期、各部分的成功 / 空 / 失败数、失败清单、**每年覆盖率**
  - `coverage(stocks: pd.DataFrame, daily: pd.DataFrame, reports: dict[str, pd.DataFrame]) -> pd.DataFrame`：行是年份，列是"当年上市股票数""有日线的""每张财务表有数据的"

- [ ] **步骤 1：写失败的测试**

```python
def test_coverage_counts_listed_stocks_per_year():
    # 3 只股票：A 2009 上市一直在，B 2012 退市，C 2015 上市；B 的财务只有到 2010
    cov = coverage(stocks, daily, {"income": income})
    assert cov.loc[2011, "listed"] == 2 and cov.loc[2011, "income"] == 1

def test_pack_roundtrip(tmp_path):
    # 用任务 1、2 测试里的假数据跑 pack；解包后读 daily.parquet，行数、dtype、instrument 写法正确；meta.json 里有 coverage
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 在 `data/external/README.md` 写用户操作步骤：装依赖 → `python scripts/collect_full_a.py --part all` → 在 GitHub 新建 Release（tag 用 `full-a-YYYYMMDD`）并上传 tar → 把 tag 告诉 Claude；注明仓库是公开的，上传的数据也会公开
- [ ] **步骤 6：** 提交，并推送、开 PR，**请用户开始采集**（后面的任务 4–7 可以先用假数据做）

### 任务 4：云端下载

**文件：**
- 新建：`scripts/download_full_a.py`
- 测试：`tests/test_download_full_a.py`

**接口：**
- 消费：任务 3 的 tar 结构
- 产出：
  - `FULL_A_DIR`（`qt/config.py`，环境变量 `QT_FULL_A_DIR`，默认 `~/.qlib/qt_full_a`）
  - `release_asset_url(tag: str, repo: str = "928460495/Quantitative-Trading") -> str`：用 GitHub API 查这个 tag 下以 `full_a_` 开头、`.tar` 结尾的附件
  - `install(archive: Path, target: Path, tag: str) -> None`：解包到 `target`，写 `target / ".version"`（内容为 tag）；只接受 tar 里的普通文件，路径不能跳出 `target`

- [ ] **步骤 1：写失败的测试**

```python
def test_install_rejects_path_traversal(tmp_path):
    # tar 里有 "../evil.txt"：抛 ValueError，target 外面没有新文件

def test_install_writes_version(tmp_path):
    install(good_tar, tmp_path / "full_a", "full-a-20261027")
    assert (tmp_path / "full_a" / ".version").read_text() == "full-a-20261027"
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 5：涨跌停

**文件：**
- 新建：`qt/limits.py`
- 测试：`tests/test_limits.py`

**接口：**
- 产出：
  - `CHINEXT_REFORM = pd.Timestamp("2020-08-24")`
  - `limit_pct(instrument: pd.Index, date: pd.DatetimeIndex, is_st: pd.Series) -> pd.Series`：主板 10%、主板 ST 5%；创业板（`SZ300`、`SZ301`）2020-08-24 前 10%（ST 5%）、之后 20%（含 ST）；科创板（`SH688`、`SH689`）20%
  - `limit_prices(preclose: pd.Series, pct: pd.Series) -> tuple[pd.Series, pd.Series]`：`preclose × (1 ± pct)` 四舍五入到分（用 `Decimal` 的 ROUND_HALF_UP 规则，不用 Python 的银行家舍入）
  - `limit_flags(close: pd.Series, preclose: pd.Series, pct: pd.Series) -> tuple[pd.Series, pd.Series]`：收盘价 ≥ 涨停价为涨停，≤ 跌停价为跌停（容差 0.001 元）

- [ ] **步骤 1：写失败的测试**

```python
def test_limit_pct_by_board_and_date():
    # SH600000 → 0.10；SH600000 ST → 0.05；SZ300750 2020-08-21 → 0.10、2020-08-24 → 0.20；SZ300xxx ST 2020-08-24 → 0.20；SH688001 → 0.20

def test_limit_price_rounding_half_up():
    up, down = limit_prices(pd.Series([10.05]), pd.Series([0.10]))
    assert up.iloc[0] == 11.06 and down.iloc[0] == 9.05   # 11.055 → 11.06，9.045 → 9.05

def test_ex_dividend_day_uses_reference_preclose():
    # 前一天收盘 10.00，除息后参考价 9.50，当天收盘 10.45：按 9.50 算是涨停（10.45），按 10.00 算不是
    at_up, _ = limit_flags(pd.Series([10.45]), pd.Series([9.50]), pd.Series([0.10]))
    assert at_up.iloc[0]
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 6：读取与股票池

**文件：**
- 新建：`qt/full_a.py`
- 修改：`qt/universe.py`
- 测试：`tests/test_full_a.py`、`tests/test_universe.py`

**接口：**
- 消费：任务 4 的 `FULL_A_DIR`，任务 5 的三个函数
- 产出：
  - `load_daily(columns: list[str], start: str, end: str, directory: Path = FULL_A_DIR) -> pd.DataFrame`：以 `(datetime, instrument)` 为索引；先 `check_range(start, end)`
  - `LIQUIDITY_CUT = 0.2`
  - `full_a_masks(df: pd.DataFrame, daily: pd.DataFrame, listed: pd.Series) -> tuple[pd.Series, pd.Series]`：返回 `(can_buy, can_sell)`，索引同 `df`（信号日 T）
    - 能买：入场日（T+1）不是 ST、交易状态正常、收盘没涨停；信号日上市满 365 天；不是北交所；信号日 20 日平均成交额不在当天最小的 20%
    - 能卖：入场日交易状态正常、收盘没跌停
    - `df` 需要 `FILTER_EXPRS` 的列加 `"_amount20": "Mean($amount,20)"`；`daily` 是 `load_daily(["preclose", "close", "tradestatus", "isST"], ...)`
  - `listing_dates_full_a(directory: Path = FULL_A_DIR) -> pd.Series`：从 `stocks.parquet` 的 `ipo_date` 取上市日，索引为 instrument
  - `float_shares(daily: pd.DataFrame) -> pd.Series`：`volume / (turn / 100)`（baostock 的 volume 单位是股、turn 单位是 %）；`turn` 为 0 或缺失的日子用该股前一个有效值。给规模因子和中性化用，补上巨潮没有的退市股股本
  - `load_index(symbol: str, start: str, end: str, directory: Path = FULL_A_DIR) -> pd.Series`：指数日收盘价，先 `check_range`

- [ ] **步骤 1：写失败的测试**

```python
def test_load_daily_respects_test_lock(tmp_path):
    with pytest.raises(TestSegmentLockedError):
        load_daily(["close"], "2021-01-01", "2022-06-30", tmp_path)

def test_st_on_entry_day_blocks_buy():
    # T 日正常、T+1 被 ST：T 日 can_buy 为 False

def test_bottom_liquidity_excluded():
    # 10 只股票，成交额最小的 2 只 can_buy 为 False

def test_float_shares_from_turnover():
    # volume 1_000_000 股、turn 2.0（%）→ 50_000_000 股；下一天停牌 turn 为 0 → 沿用 50_000_000

def test_limit_up_uses_real_limit_for_chinext():
    # 创业板 2021 年 T+1 涨 15%：不是涨停，能买（原来 9.5% 的统一规则会误判成买不进）
```

- [ ] **步骤 2–4：** 失败 → 实现 → 通过
- [ ] **步骤 5：** 提交

### 任务 7：按发布日期对齐的财务数据

**拿到真实数据后再做**：东方财富接口的原始列名要看真实文件才能定。

**文件：**
- 新建：`qt/fundamentals.py`
- 修改：`docs/DATA_NOTES.md`（列名对照、覆盖率、发现的问题）
- 测试：`tests/test_fundamentals.py`

**接口：**
- 产出：
  - `COLUMN_MAP: dict[str, dict[str, str]]`：每张表从原始列名到统一列名（`instrument, report_date, notice_date, net_profit, revenue, op_cash_flow, total_equity, total_assets, total_liab, gross_profit`），在步骤 1 看过真实文件后写定
  - `tidy_reports(raw: dict[str, pd.DataFrame]) -> pd.DataFrame`：每个（股票，报告期）一行；多次公告只留最早的 `notice_date`
  - `single_quarter(df: pd.DataFrame, col: str) -> pd.Series`、`ttm(df: pd.DataFrame, col: str) -> pd.Series`：从年初至今的累计值算单季、滚动 4 季
  - `yoy(df: pd.DataFrame, col: str) -> pd.Series`：单季同比，基数 ≤ 0 时为 NaN
  - `align_to_days(records: pd.DataFrame, cols: list[str], index: pd.MultiIndex, calendar: pd.DatetimeIndex) -> pd.DataFrame`：`notice_date` 之后的第一个交易日起可用，按 `merge_asof` 往后填

- [ ] **步骤 1：** 读真实的 `reports/*.parquet`，写 `COLUMN_MAP` 和 `DATA_NOTES.md` 的列名对照、覆盖率表
- [ ] **步骤 2：写失败的测试**

```python
def test_annual_report_not_visible_before_notice():
    # 2020 年报 2021-04-28（周三）公告：2021-04-28 当天不可用，2021-04-29 起可用
def test_restatement_keeps_first_notice():
def test_single_quarter_and_ttm_from_cumulative():
    # 累计净利润 Q1=10、H1=25、Q3=45、FY=70 → 单季 10、15、20、25；下一年 Q1=12 → TTM = 15+20+25+12 = 72
def test_yoy_nan_when_base_not_positive():
```

- [ ] **步骤 3–5：** 失败 → 实现 → 通过
- [ ] **步骤 6：** 核对 baostock 日线估值是否按发布日期：用 `close / peTTM` 反推每股收益，统计它变化的日期落在东方财富公告日之前的比例，写进 `docs/DATA_NOTES.md`。**事先写定的规则**：超过 5% 的变化发生在公告日之前，就不用 baostock 的估值，改用 `qt/fundamentals.py` 自己算
- [ ] **步骤 7：** 提交，推送，PR 说明里附覆盖率表
