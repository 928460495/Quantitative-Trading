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
