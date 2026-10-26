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
