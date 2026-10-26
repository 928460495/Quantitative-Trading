"""用一个假的 AkShare 测采集脚本的逻辑：重试、失败记录、断点续跑。不联网。"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collect_external", ROOT / "scripts" / "collect_external.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class FakeAk:
    __version__ = "fake"

    def __init__(self, flaky=(), broken=()):
        self.calls: dict[str, int] = {}
        self.flaky, self.broken = set(flaky), set(broken)

    def stock_industry_clf_hist_sw(self):
        return pd.DataFrame({"symbol": ["600000", "000001"], "start_date": ["2011-01-01", "2011-01-01"],
                             "industry_code": ["480301", "480101"], "update_time": [None, None]})

    def stock_share_change_cninfo(self, symbol, start_date, end_date):
        self.calls[symbol] = self.calls.get(symbol, 0) + 1
        if symbol in self.broken or (symbol in self.flaky and self.calls[symbol] == 1):
            raise ConnectionError("boom")
        return pd.DataFrame({"公告日期": ["2015-01-10"], "变动日期": ["2015-01-05"], "总股本": [100.0],
                             "人民币普通股": [60.0], "已流通股份": [60.0]})


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "EXTERNAL_DIR", tmp_path / "external")
    monkeypatch.setattr(collect, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "external").mkdir()
    return tmp_path


def test_industry_written(dirs, capsys):
    collect.collect_industry(FakeAk())
    df = pd.read_csv(dirs / "external" / collect.INDUSTRY_FILE, dtype=str)
    assert df["symbol"].tolist() == ["000001", "600000"]
    assert "一级行业" in capsys.readouterr().out


def test_shares_retry_failure_and_resume(dirs):
    ak = FakeAk(flaky={"000001"}, broken={"000002"})
    shares, failed = collect.collect_shares(ak, ["000001", "000002", "600000"], pause=0, retries=2)
    assert ak.calls == {"000001": 2, "000002": 2, "600000": 1}
    assert set(failed) == {"000002"}
    assert sorted(shares["symbol"].unique()) == ["000001", "600000"]

    # 重跑：已缓存的不再请求，只补失败的
    ak2 = FakeAk()
    shares, failed = collect.collect_shares(ak2, ["000001", "000002", "600000"], pause=0, retries=2)
    assert ak2.calls == {"000002": 1}
    assert not failed
    assert sorted(shares["symbol"].unique()) == ["000001", "000002", "600000"]


SW_ROWS = {"股票代码": ["000001", "600000"], "计入日期": ["2011-01-01", "2014-06-30"],
           "行业代码": ["480101", "480301"], "更新日期": ["2021-07-30", "2021-07-30"]}


def test_industry_from_local_file(dirs, tmp_path):
    path = tmp_path / "StockClassifyUse_stock.csv"
    pd.DataFrame(SW_ROWS).to_csv(path, index=False)
    df = collect.collect_industry(None, file=path)
    assert df["symbol"].tolist() == ["000001", "600000"]
    assert df["industry_code"].tolist() == ["480101", "480301"]  # 代码前面的 0 不能丢
    assert (dirs / "external" / collect.INDUSTRY_FILE).exists()


def test_industry_from_local_xlsx(dirs, tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "StockClassifyUse_stock.xlsx"
    pd.DataFrame(SW_ROWS).to_excel(path, index=False)
    df = collect.collect_industry(None, file=path)
    assert df["symbol"].tolist() == ["000001", "600000"]


def test_industry_file_with_wrong_columns(dirs, tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"代码": ["000001"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="缺少列"):
        collect.collect_industry(None, file=path)


def test_industry_download_failure_does_not_stop(dirs, capsys):
    class Broken(FakeAk):
        def stock_industry_clf_hist_sw(self):
            raise ConnectionError("ssl")

    assert collect.collect_industry(Broken(), retries=1, pause=0) is None
    out = capsys.readouterr().out
    assert "StockClassifyUse_stock.xls" in out and "--industry-file" in out


def test_empty_cninfo_response_counts_as_no_records(dirs):
    class Delisted(FakeAk):
        def stock_share_change_cninfo(self, symbol, start_date, end_date):
            self.calls[symbol] = self.calls.get(symbol, 0) + 1
            raise KeyError("公告日期")  # AkShare 处理空表时的报错

    ak = Delisted()
    shares, failed = collect.collect_shares(ak, ["000024"], pause=0, retries=3)
    assert ak.calls == {"000024": 1}  # 不重试
    assert not failed
    assert (dirs / "cache" / "000024.csv").exists()  # 记为无记录，重跑时不再请求
