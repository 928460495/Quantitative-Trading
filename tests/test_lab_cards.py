"""实验室：假设卡、预算、基线（docs/FRAMEWORK.md 第 4、7 节）。"""

import subprocess

import pytest
import yaml

from qt.lab import cards as cd
from qt.lab import procedure as pc

GOOD = {
    "id": "H001", "parent": None, "family": "问题股排除", "kind": "防雷型",
    "mechanism": "便宜又亏损的公司在退市变真之后更可能继续跌（Novy-Marx 2013；2020-12 退市新规）",
    "change": "扣非滚动四季为负的不买，持有的下个调仓日卖", "prediction": {"metric": "三.大亏只数", "direction": "降"},
    "falsification": "阶段三大亏只数没有减少三成，或阶段三收益变差", "variant": {"exclude": ["is_st", "loss_small", "cheap", "deduct_loss"]},
    "seen_before": "3d 在 2019–2021 的逐只盈亏", "status": "已登记",
}


def _write(d, card):
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{card['id']}.yaml"
    path.write_text(yaml.safe_dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_good_card_loads(tmp_path):
    assert cd.load_card(_write(tmp_path, GOOD))["variant"] == GOOD["variant"]


@pytest.mark.parametrize("change, message", [
    ({"kind": "随便型"}, "kind"), ({"status": "做完了"}, "status"),
    ({"prediction": {"metric": "三.大亏只数"}}, "prediction"), ({"prediction": {"metric": "三.大亏只数", "direction": "好"}}, "prediction"),
    ({"prediction": {"metric": "大亏只数", "direction": "降"}}, "写法"), ({"variant": {}}, "variant"),
    ({"mechanism": "  "}, "mechanism"), ({"falsification": None}, "falsification"), ({"seen_before": ""}, "seen_before"),
])
def test_bad_cards_are_rejected(tmp_path, change, message):
    with pytest.raises(cd.CardError, match=message):
        cd.load_card(_write(tmp_path, {**GOOD, **change}))


def test_missing_field_and_mismatched_id(tmp_path):
    card = {k: v for k, v in GOOD.items() if k != "falsification"}
    with pytest.raises(cd.CardError, match="缺字段"):
        cd.load_card(_write(tmp_path, card))
    path = _write(tmp_path / "x", GOOD)
    path.rename(path.with_name("H002.yaml"))
    with pytest.raises(cd.CardError, match="不一致"):
        cd.load_card(path.with_name("H002.yaml"))


def test_is_committed_tracks_git_state(tmp_path):
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    path = _write(tmp_path, GOOD)
    assert not cd.is_committed(path)                                          # 没进仓库
    run("add", path.name)
    assert not cd.is_committed(path)                                          # 只是暂存，没提交
    run("commit", "-q", "-m", "x")
    assert cd.is_committed(path)
    path.write_text(path.read_text() + "\n# 改了一下\n")
    assert not cd.is_committed(path)                                          # 提交后又改了


def _cards(*specs):
    return [{**GOOD, "id": i, "family": f, "status": s} for i, f, s in specs]


def test_budget_per_card_and_status():
    cards = _cards(("H001", "甲", "已登记"))
    run = lambda c, n: [{"card": c, "run": i + 1} for i in range(n)]
    cd.check_budget(cards[0], cards, run("H001", 5))
    with pytest.raises(cd.CardError, match="最多 6 次"):
        cd.check_budget(cards[0], cards, run("H001", 6))
    cd.check_budget(cards[0], cards, run("H002", 9))                         # 别的卡的次数不算
    done = {**cards[0], "status": "未采纳"}
    with pytest.raises(cd.CardError, match="已经有结论"):
        cd.check_budget(done, [done], [])


def test_budget_per_generation_and_family():
    many = _cards(*[(f"H{i:03d}", f"方向{i}", "已登记") for i in range(1, 14)])
    with pytest.raises(cd.CardError, match="最多 12 张"):
        cd.check_budget(many[0], many, [])
    cd.check_budget(many[0], many[:12], [])
    fam = _cards(("H001", "甲", "已登记"), ("H002", "甲", "已登记"), ("H003", "甲", "已登记"), ("H004", "甲", "已登记"))
    with pytest.raises(cd.CardError, match="最多 3 张"):
        cd.check_budget(fam[0], fam, [])


def test_cooldown_after_two_rejections_in_a_row():
    two_no = _cards(("H001", "甲", "未采纳"), ("H002", "甲", "未采纳"), ("H003", "甲", "已登记"))
    with pytest.raises(cd.CardError, match="冷却"):
        cd.check_budget(two_no[2], two_no, [])
    mixed = _cards(("H001", "甲", "未采纳"), ("H002", "甲", "已采纳"), ("H003", "甲", "已登记"))
    cd.check_budget(mixed[2], mixed, [])
    other = _cards(("H001", "乙", "未采纳"), ("H002", "乙", "未采纳"), ("H003", "甲", "已登记"))
    cd.check_budget(other[2], other, [])                                      # 别的方向不影响
    abandoned = _cards(("H001", "甲", "未采纳"), ("H002", "甲", "放弃"), ("H003", "甲", "已登记"))
    cd.check_budget(abandoned[2], abandoned, [])                              # 放弃的不算一次没采纳


def test_baseline_and_candidate(tmp_path):
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump({"procedure": pc.PLAN3D.to_dict()}, allow_unicode=True), encoding="utf-8")
    base = cd.load_baseline(path)
    assert base == pc.PLAN3D
    cand = cd.candidate(GOOD, base)
    assert cand.exclude == ("cheap", "deduct_loss", "is_st", "loss_small") and cand.replace(exclude=list(base.exclude)) == base
    with pytest.raises(cd.CardError, match="variant"):
        cd.candidate({**GOOD, "variant": {"n_holdings": 30}}, base)


def test_runs_append_only(tmp_path):
    path = tmp_path / "runs.jsonl"
    assert cd.load_runs(path) == []
    cd.append_run(path, {"card": "H001", "x": 1.5})
    cd.append_run(path, {"card": "H001", "x": "中文"})
    assert cd.load_runs(path) == [{"card": "H001", "x": 1.5}, {"card": "H001", "x": "中文"}]


def test_void_runs_do_not_use_budget_but_keep_their_numbers():
    runs = [{"card": "H001", "run": 1, "adopt": False}, {"card": "H001", "void": 1, "reason": "缓存钥匙漏了 missing"},
            {"card": "H002", "run": 1, "adopt": True}]
    assert cd.valid_runs(runs, "H001") == [] and len(cd.valid_runs(runs, "H002")) == 1
    assert cd.next_run_number(runs, "H001") == 2                               # 作废的编号不复用
    cards = _cards(("H001", "甲", "已登记"))
    full = [{"card": "H001", "run": i, "adopt": False} for i in range(1, 7)]
    with pytest.raises(cd.CardError, match="最多 6 次"):
        cd.check_budget(cards[0], cards, full)
    cd.check_budget(cards[0], cards, full + [{"card": "H001", "void": 3, "reason": "x"}])
    with pytest.raises(cd.CardError, match="没有第 9 次"):
        cd.void_record(full, "H001", 9, "x")
    rec = cd.void_record(full, "H001", 2, "程序错误")
    assert rec["card"] == "H001" and rec["void"] == 2 and rec["reason"] == "程序错误" and "time" in rec
    with pytest.raises(cd.CardError, match="写明原因"):
        cd.void_record(full, "H001", 2, "  ")
