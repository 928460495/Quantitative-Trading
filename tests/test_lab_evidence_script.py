"""实验室：便宜证据的脚本（scripts/lab_evidence.py）。小假数据。"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from qt.lab import procedure as pc

from lab_helpers import dev3d

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lab_evidence", ROOT / "scripts" / "lab_evidence.py")
le = importlib.util.module_from_spec(spec)
sys.modules["lab_evidence"] = le
spec.loader.exec_module(le)


def test_build_and_write(tmp_path):
    dev, _ = dev3d()
    cols = dev["close"].columns
    dev["flags"]["is_st"].loc["2019-06-03":, cols[5]] = True
    dev["flags"]["deduct_loss"].loc["2019-01-02":, cols[5]] = True
    dev["flags"]["is_st"].loc["2018-09-03":, cols[9]] = True                   # 另一只在阶段二戴帽，事前没有亏损标记
    secs = le.build(dev, pc.PLAN3D)
    assert list(secs) == ["顶端各档（现行基线，含排除）", "前50名里带标记的股票（不做任何排除的打分）", "第一次戴 ST 前后（按戴帽所在阶段）"]
    st = secs["第一次戴 ST 前后（按戴帽所在阶段）"]
    assert st.loc["二", "事件数"] == 1 and st.loc["三", "事件数"] == 1 and st.loc["全段", "事件数"] == 2
    assert st.loc["二", "deduct_loss 事前亮着的比例"] == 0.0 and st.loc["全段", "deduct_loss 事前亮着的比例"] == 0.5
    assert st.loc["三", "deduct_loss 事前亮着的比例"] == 1.0 and st.loc["三", "neg_equity 事前亮着的比例"] == 0.0
    split = secs["前50名里带标记的股票（不做任何排除的打分）"]
    assert ("三", "deduct_loss") in split.index and ("全段", "small30") in split.index
    le.write(secs, tmp_path / "E.md", "测试")
    text = (tmp_path / "E.md").read_text()
    assert "不计试验次数" in text and "第一次戴 ST" in text


def test_st_summary_empty():
    assert le.st_summary(pd.DataFrame()).empty
