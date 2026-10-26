import numpy as np
import pytest

from qt.trials import config_hash, count_unique, log_trial


def test_identical_configs_count_once(tmp_path):
    path = tmp_path / "trials.jsonl"
    log_trial("single_factor", {"factor": "a", "rule": "30/90"}, {"t": 3.1}, path)
    log_trial("single_factor", {"rule": "30/90", "factor": "a"}, {"t": 3.1}, path)
    log_trial("single_factor", {"factor": "a", "rule": "30/60"}, {"t": 2.0}, path)
    log_trial("combination", {"method": "equal"}, {"t": 4.0}, path)
    assert count_unique(path) == 3
    assert count_unique(path, kind="single_factor") == 2
    assert len(path.read_text().splitlines()) == 4


def test_hash_ignores_key_order():
    assert config_hash("k", {"a": 1, "b": 2}) == config_hash("k", {"b": 2, "a": 1})


def test_load_unique_keeps_last_record_per_config(tmp_path):
    from qt.trials import load_unique

    path = tmp_path / "trials.jsonl"
    log_trial("combination", {"method": "equal"}, {"net_ann": 0.01}, path)
    log_trial("combination", {"method": "equal"}, {"net_ann": 0.02}, path)
    log_trial("combination", {"method": "icir"}, {"net_ann": 0.03}, path)
    rows = load_unique(path)
    assert len(rows) == 2
    assert {r["params"]["method"]: r["metrics"]["net_ann"] for r in rows} == {"equal": 0.02, "icir": 0.03}


def test_expected_max_sharpe_grows_with_trials():
    from qt.trials import expected_max_sharpe

    assert expected_max_sharpe(1, 1.0) == 0.0
    a, b = expected_max_sharpe(10, 1.0), expected_max_sharpe(1000, 1.0)
    assert 0 < a < b
    # 1000 次标准正态的最大值期望约 3.24
    assert b == pytest.approx(3.24, abs=0.05)
    assert expected_max_sharpe(1000, 4.0) == pytest.approx(2 * b)


def test_deflated_sharpe_single_trial_is_probabilistic_sharpe():
    from scipy.stats import norm

    from qt.trials import deflated_sharpe

    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 750)
    out = deflated_sharpe(r, n_trials=1, sharpe_var=0.0)
    sr = r.mean() / r.std(ddof=1)
    assert out["sharpe"] == pytest.approx(sr)
    # 正态收益时分母约为 sqrt(1 + sr²/2)
    assert out["dsr"] == pytest.approx(norm.cdf(sr * np.sqrt(749) / np.sqrt(1 + sr**2 / 2)), abs=0.01)


def test_deflated_sharpe_drops_with_more_trials():
    from qt.trials import deflated_sharpe

    r = np.random.default_rng(1).normal(0.001, 0.01, 750)
    few = deflated_sharpe(r, n_trials=5, sharpe_var=0.03**2)["dsr"]
    many = deflated_sharpe(r, n_trials=500, sharpe_var=0.03**2)["dsr"]
    assert many < few
