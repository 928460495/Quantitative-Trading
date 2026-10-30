import numpy as np
import pandas as pd
import pytest

from qt import ops
from qt.formula import FormulaError, evaluate, parse

IDX = pd.bdate_range("2019-01-02", periods=30)
rng = np.random.default_rng(0)
ENV = {name: pd.DataFrame(rng.uniform(5, 15, (30, 4)), index=IDX, columns=list("ABCD"))
       for name in ["open", "high", "low", "close", "vwap", "volume", "amount"]}
ENV["returns"] = ENV["close"].pct_change()


def ev(text, dialect="gtja"):
    return evaluate(text, ENV, dialect=dialect)


def test_precedence_and_power():
    pd.testing.assert_frame_equal(ev("CLOSE + OPEN * 2 ^ 2"), ENV["close"] + ENV["open"] * 4)
    pd.testing.assert_frame_equal(ev("-CLOSE ^ 2"), -(ENV["close"] ** 2))
    pd.testing.assert_frame_equal(ev("2 ^ 3 ^ 2 * CLOSE"), 512 * ENV["close"])   # 乘方右结合


def test_ternary_nested_and_equals():
    c, o = ENV["close"], ENV["open"]
    expect = pd.DataFrame(np.where(c > o, 1.0, np.where(c == o, 0.0, -1.0)), index=IDX, columns=list("ABCD"))
    pd.testing.assert_frame_equal(ev("CLOSE>OPEN?1:(CLOSE=OPEN?0:-1)"), expect)


def test_logic_ops():
    c, o = ENV["close"], ENV["open"]
    got = ev("(CLOSE>OPEN && CLOSE>10) || CLOSE<6")
    expect = (((c > o) & (c > 10)) | (c < 6)).astype(float)
    pd.testing.assert_frame_equal(got, expect)


def test_functions_map_to_ops_case_insensitive():
    pd.testing.assert_frame_equal(ev("TSRANK(CLOSE, 5)"), ops.ts_rank(ENV["close"], 5))
    pd.testing.assert_frame_equal(ev("ts_rank(close, 5.9)", "wq"), ops.ts_rank(ENV["close"], 5))   # 非整数窗口取整
    pd.testing.assert_frame_equal(ev("RANK(CLOSE)"), ops.cs_rank(ENV["close"]))


def test_max_min_second_arg_rule():
    c = ENV["close"]
    pd.testing.assert_frame_equal(ev("MAX(CLOSE, 15)"), ops.ts_max(c, 15))          # 整数 ≥ 2：时间窗口
    pd.testing.assert_frame_equal(ev("MAX(CLOSE-OPEN, 0)"), np.maximum(c - ENV["open"], 0.0))   # 0：逐元素
    pd.testing.assert_frame_equal(ev("MIN(CLOSE, OPEN)"), np.minimum(c, ENV["open"]))


def test_variables_adv_and_ret():
    pd.testing.assert_frame_equal(ev("adv20", "wq"), ops.ts_mean(ENV["amount"], 20))
    pd.testing.assert_frame_equal(ev("RET"), ENV["returns"])


def test_gtja_intermediates():
    o, h, c = ENV["open"], ENV["high"], ENV["close"]
    hd = h - h.shift(1)
    pd.testing.assert_frame_equal(ev("HD"), hd)
    tr = np.maximum(np.maximum(h - ENV["low"], (h - c.shift(1)).abs()), (ENV["low"] - c.shift(1)).abs())
    pd.testing.assert_frame_equal(ev("TR"), tr)


def test_two_dialects_same_result():
    a = ev("(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))", "wq")
    b = ev("(-1 * CORR(RANK(DELTA(LOG(VOLUME), 2)), RANK(((CLOSE - OPEN) / OPEN)), 6))", "gtja")
    pd.testing.assert_frame_equal(a, b)


def test_regbeta_with_sequence():
    pd.testing.assert_frame_equal(ev("REGBETA(MEAN(CLOSE,6),SEQUENCE(6),6)"),
                                  ops.regbeta(ops.ts_mean(ENV["close"], 6), ops.sequence(6), 6))


def test_unknown_names_raise():
    with pytest.raises(FormulaError, match="FOO"):
        ev("FOO(CLOSE, 3)")
    with pytest.raises(FormulaError, match="BAR"):
        ev("CLOSE + BAR")


def test_parse_reports_variables():
    tree = parse("CORR(VWAP, MEAN(VOLUME,60), 6) + adv20", "wq")
    assert tree.variables() == {"vwap", "volume", "adv20"}
