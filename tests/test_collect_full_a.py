"""用假的 baostock、假的东方财富接口测全 A 采集脚本：代码转换、断点续跑、失败记录、打包。不联网。"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_full_a", ROOT / "scripts" / "collect_full_a.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class FakeRS:
    """模仿 baostock 的 ResultData：next() 逐行前进，get_row_data() 取当前行。"""

    def __init__(self, fields, rows, error_code="0", fail_at=None):
        self.fields, self._rows, self.error_code, self.error_msg = fields, rows, error_code, "msg"
        self._i, self._fail_at = -1, fail_at

    def next(self):
        self._i += 1
        if self._fail_at is not None and self._i == self._fail_at:
            raise ConnectionError("连接中断")
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


DAILY_ROW = ["2015-07-01", "{code}", "10.0", "10.5", "1000000", "10500000.0", "2.0", "1", "0", "8.1", "1.2", "1.5", "6.0"]


class FakeBs:
    def __init__(self, broken=(), error_once=(), crash=(), empty=()):
        self.calls: list[str] = []
        self.logins = 0
        self.broken, self.error_once, self.crash, self.empty = set(broken), set(error_once), set(crash), set(empty)
        self._errored: set[str] = set()

    def login(self):
        self.logins += 1
        return FakeRS([], [])

    def query_stock_basic(self, code="", code_name=""):
        fields = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
        rows = [
            ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
            ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
            ["sh.000300", "沪深300", "2005-04-08", "", "2", "1"],
            ["bj.430047", "诺思兰德", "2021-11-15", "", "1", "1"],
            ["sz.000003", "PT金田A", "1991-07-03", "2002-06-14", "1", "0"],
        ]
        return FakeRS(fields, rows)

    def query_history_k_data_plus(self, code, fields, start_date=None, end_date=None, frequency="d", adjustflag="3"):
        self.calls.append(code)
        names = fields.split(",")
        if code in self.broken:
            raise ConnectionError("网络接收错误")
        if code in self.error_once and code not in self._errored:
            self._errored.add(code)
            return FakeRS(names, [], error_code="10001001")
        if code in self.empty:
            return FakeRS(names, [])
        rows = [[v.format(code=code) for v in DAILY_ROW], [v.format(code=code) for v in DAILY_ROW]]
        rows[1][0] = "2015-07-02"
        return FakeRS(names, rows, fail_at=1 if code in self.crash else None)


def test_to_instrument():
    assert collect.to_instrument("sh.600000") == "SH600000"
    assert collect.to_instrument("sz.000001") == "SZ000001"
    assert collect.to_instrument("300750") == "SZ300750"
    assert collect.to_instrument("688001") == "SH688001"
    assert collect.to_instrument("SH600000") == "SH600000"
    assert collect.to_instrument("bj.430047") == "BJ430047"
    assert collect.to_instrument("830799") == "BJ830799"


def test_stock_list_keeps_only_stocks_and_drops_bj():
    out = collect.stock_list(FakeBs())
    assert list(out.columns) == ["instrument", "name", "ipo_date", "out_date"]
    assert list(out["instrument"]) == ["SH600000", "SZ000001", "SZ000003"]
    assert out.set_index("instrument").loc["SZ000003", "out_date"] == "2002-06-14"


def test_collect_daily_resumes(tmp_path):
    bs = FakeBs(broken={"sz.000001"})
    first = collect.collect_daily(bs, ["SH600000", "SZ000001"], tmp_path, retries=3)
    assert first["SH600000"] == "ok"
    assert first["SZ000001"].startswith("failed: ")
    assert bs.calls.count("sz.000001") == 3
    df = pd.read_csv(tmp_path / "SH600000.csv")
    assert len(df) == 2 and list(df.columns) == collect.DAILY_FIELDS.split(",")

    bs2 = FakeBs()
    second = collect.collect_daily(bs2, ["SH600000", "SZ000001"], tmp_path, retries=3)
    assert bs2.calls == ["sz.000001"]
    assert second == {"SH600000": "ok", "SZ000001": "ok"}


def test_collect_daily_relogins_on_error_code(tmp_path):
    bs = FakeBs(error_once={"sh.600000"})
    status = collect.collect_daily(bs, ["SH600000"], tmp_path, retries=3)
    assert status == {"SH600000": "ok"}
    assert bs.logins == 1


def test_collect_daily_empty_is_recorded_and_not_refetched(tmp_path):
    bs = FakeBs(empty={"sz.000003"})
    assert collect.collect_daily(bs, ["SZ000003"], tmp_path) == {"SZ000003": "empty"}
    assert pd.read_csv(tmp_path / "SZ000003.csv").empty
    bs2 = FakeBs(empty={"sz.000003"})
    assert collect.collect_daily(bs2, ["SZ000003"], tmp_path) == {"SZ000003": "empty"}
    assert bs2.calls == []


def test_collect_daily_no_partial_file(tmp_path):
    bs = FakeBs(crash={"sh.600000"})
    status = collect.collect_daily(bs, ["SH600000"], tmp_path, retries=2)
    assert status["SH600000"].startswith("failed: ")
    assert list(tmp_path.iterdir()) == []


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """两页数据的东方财富接口；empty 里的报告期返回 result 为 null。"""

    def __init__(self, empty=(), broken=()):
        self.calls: list[dict] = []
        self.empty, self.broken = set(empty), set(broken)

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        flt = params["filter"]
        if any(d in flt for d in self.broken):
            raise ConnectionError("连接被重置")
        if any(d in flt for d in self.empty):
            return FakeResponse({"result": None, "success": False, "message": "返回数据为空"})
        page = int(params["pageNumber"])
        rows = [{"SECURITY_CODE": "600000", "NOTICE_DATE": "2016-03-30 00:00:00", "PARENT_NETPROFIT": 1.0}] * (2 if page == 1 else 1)
        return FakeResponse({"result": {"pages": 2, "data": rows}, "success": True})


class FakeAkIndex:
    def __init__(self, starts):
        self.starts = starts

    def stock_zh_index_hist_csindex(self, symbol, start_date, end_date):
        if symbol not in self.starts:
            raise KeyError("data")
        dates = pd.bdate_range(self.starts[symbol], periods=3)
        return pd.DataFrame({"日期": dates.date, "指数代码": symbol, "收盘": [1000.0, 1001.0, 1002.0]})


def test_report_dates():
    assert collect.report_dates("2008-03-31", "2008-12-31") == ["2008-03-31", "2008-06-30", "2008-09-30", "2008-12-31"]
    assert collect.report_dates("2008-03-31", "2008-08-15") == ["2008-03-31", "2008-06-30"]


def test_collect_reports_reads_all_pages(tmp_path):
    session = FakeSession()
    status = collect.collect_reports(session, ["income"], ["2015-12-31"], tmp_path)
    df = pd.read_csv(tmp_path / "income" / "2015-12-31.csv")
    assert len(df) == 3 and "NOTICE_DATE" in df.columns
    assert status == {"income/2015-12-31": "ok"}
    first = session.calls[0]
    assert first["reportName"] == "RPT_DMSK_FN_INCOME" and first["columns"] == "ALL"
    assert "(REPORT_DATE='2015-12-31')" in first["filter"]
    assert collect.collect_reports(FakeSession(), ["income"], ["2015-12-31"], tmp_path) == {"income/2015-12-31": "ok"}


def test_collect_reports_yjbb_uses_its_own_date_field(tmp_path):
    session = FakeSession()
    collect.collect_reports(session, ["yjbb"], ["2015-12-31"], tmp_path)
    assert session.calls[0]["reportName"] == "RPT_LICO_FN_CPD"
    assert "(REPORTDATE='2015-12-31')" in session.calls[0]["filter"]


def test_collect_reports_empty_period(tmp_path):
    session = FakeSession(empty={"2008-03-31"})
    status = collect.collect_reports(session, ["income"], ["2008-03-31"], tmp_path, retries=3)
    assert status == {"income/2008-03-31": "empty"}
    assert len(session.calls) == 1
    assert not (tmp_path / "income" / "2008-03-31.csv").exists()


def test_collect_reports_failure_is_recorded(tmp_path):
    session = FakeSession(broken={"2016-03-31"})
    status = collect.collect_reports(session, ["balance"], ["2016-03-31"], tmp_path, retries=2, pause=0.0)
    assert status["balance/2016-03-31"].startswith("failed: ")
    assert len(session.calls) == 2


def test_collect_indexes_records_short_history(tmp_path):
    status = collect.collect_indexes(FakeAkIndex({"H00300": "2013-01-07", "000985": "2009-01-05"}), tmp_path,
                                     start="20090101", end="20090110")
    assert status["H00300"].startswith("ok 2013-01-07")
    df = pd.read_csv(tmp_path / "index" / "H00300.csv")
    assert list(df.columns) == ["date", "close"] and len(df) == 3


def test_collect_indexes_failure_is_recorded(tmp_path):
    status = collect.collect_indexes(FakeAkIndex({"000985": "2009-01-05"}), tmp_path, start="20090101")
    assert status["H00300"].startswith("failed: ")
    assert status["000985"].startswith("ok ")


def _daily_frame(instrument, dates):
    code = collect.to_bs_code(instrument)
    return pd.DataFrame([[d, code, "10.0", "10.5", "1000000", "10500000.0", "2.0", "1", "0", "8.1", "", "1.5", "6.0"]
                         for d in dates], columns=collect.DAILY_FIELDS.split(","))


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "daily").mkdir(parents=True)
    pd.DataFrame({"instrument": ["SH600000", "SZ000002", "SZ300750"], "name": ["A", "B", "C"],
                  "ipo_date": ["1999-11-10", "2005-01-04", "2015-03-02"],
                  "out_date": ["", "2012-06-29", ""]}).to_csv(raw / "stocks.csv", index=False)
    _daily_frame("SH600000", ["2010-06-01", "2011-06-01", "2015-06-01"]).to_csv(raw / "daily" / "SH600000.csv", index=False)
    _daily_frame("SZ000002", ["2010-06-01", "2011-06-01"]).to_csv(raw / "daily" / "SZ000002.csv", index=False)
    _daily_frame("SZ300750", ["2015-06-01"]).to_csv(raw / "daily" / "SZ300750.csv", index=False)
    (raw / "reports" / "income").mkdir(parents=True)
    for d, codes in {"2010-12-31": ["600000", "000002"], "2011-12-31": ["600000"], "2015-12-31": ["600000", "300750"]}.items():
        pd.DataFrame({"SECURITY_CODE": codes, "NOTICE_DATE": f"{d[:4]}-04-20 00:00:00",
                      "PARENT_NETPROFIT": 1.0}).to_csv(raw / "reports" / "income" / f"{d}.csv", index=False)
    (raw / "index").mkdir()
    pd.DataFrame({"date": ["2010-06-01", "2010-06-02"], "close": [3000.0, 3010.0]}).to_csv(raw / "index" / "H00300.csv", index=False)
    (raw / "status.json").write_text('{"daily": {"SH600000": "ok", "SZ000003": "failed: x"}}')
    return raw


def test_coverage_counts_listed_stocks_per_year(raw_dir):
    stocks, daily, reports = collect.load_raw(raw_dir)
    cov = collect.coverage(stocks, daily, reports)
    assert cov.loc[2011, "listed"] == 2 and cov.loc[2011, "daily"] == 2 and cov.loc[2011, "income"] == 1
    assert cov.loc[2015, "listed"] == 2 and cov.loc[2015, "income"] == 2


def test_pack_roundtrip(raw_dir, tmp_path):
    import json
    import tarfile

    path = collect.pack(raw_dir, tmp_path / "out", collected_on="20261027")
    assert path.name == "full_a_20261027.tar"
    with tarfile.open(path) as tar:
        names = set(tar.getnames())
        tar.extractall(tmp_path / "x", filter="data")
    assert {"stocks.parquet", "daily.parquet", "reports/income.parquet", "index.parquet", "meta.json"} <= names
    daily = pd.read_parquet(tmp_path / "x" / "daily.parquet")
    assert len(daily) == 6 and set(daily["instrument"]) == {"SH600000", "SZ000002", "SZ300750"}
    assert str(daily["date"].dtype).startswith("datetime64")
    assert daily["close"].dtype == "float32" and daily["isST"].dtype == "int8" and daily["tradestatus"].dtype == "int8"
    assert daily["pbMRQ"].isna().all()  # 空字符串读成缺失
    income = pd.read_parquet(tmp_path / "x" / "reports" / "income.parquet")
    assert {"instrument", "report_date", "SECURITY_CODE", "NOTICE_DATE"} <= set(income.columns)
    index = pd.read_parquet(tmp_path / "x" / "index.parquet")
    assert list(index.columns) == ["symbol", "date", "close"]
    meta = json.loads((tmp_path / "x" / "meta.json").read_text())
    assert meta["collected_on"] == "20261027"
    assert meta["status"]["daily"] == {"ok": 1, "failed": 1}
    assert meta["failures"]["daily"] == {"SZ000003": "failed: x"}
    assert meta["coverage"]["2011"]["income"] == 1
    assert not any(p.name.startswith(".") or p.name == "staging" for p in (tmp_path / "out").iterdir())
