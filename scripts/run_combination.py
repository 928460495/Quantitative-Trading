"""因子合成、验证段比较、Qlib 正式回测，写出 reports/combination/<market>_valid/。

用法：
    python scripts/run_combination.py              # 沪深300：训练段估权重，验证段比较
    python scripts/run_combination.py --refresh    # 不用 data/cache/ 里缓存的因子分数，重新取数

步骤（规则见 docs/DECISIONS.md 2026-10-26）：
1. 读第 4 步的最终名单 reports/screening/<market>_train_neutral/selected.csv
2. 训练段、验证段分别取数，算中性化、按方向调整后的分数
3. 训练段估权重：等权 → ICIR → 最大化 ICIR → 线性回归 → LightGBM
4. 验证段按缓冲 30/90、排雷 20% 模拟，逐级和当前方案比，明显赢了才升级
5. 选定方案上试信号平滑 3、5 天；排雷 10%、不排雷作对照
6. 对照基准 Alpha158 + LightGBM，同样的标签、切分、参数和交易规则
7. Qlib 正式回测：选定方案、成本翻倍、开盘价成交、TopkDropout、对照基准
8. 通缩夏普比率，最终测试四条标准在验证段上预演
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qt import combine  # noqa: E402
from qt.backtest import ACCOUNT, report_metrics, run_backtest  # noqa: E402
from qt.config import DATA_DIR, MAX_FORWARD_DAYS, segment  # noqa: E402
from qt.data import drop_embargo, load_features  # noqa: E402
from qt.evaluate import TRADING_DAYS, ic_series, ic_stats, newey_west_t, simulation_summary  # noqa: E402
from qt.factors import FACTOR_SETS, MAIN_HORIZON, alpha158, forward_return_expr  # noqa: E402
from qt.pipeline import EXPOSURE_EXPRS, exposures, neutral_score  # noqa: E402
from qt.preprocess import standardize  # noqa: E402
from qt.simulate import PRIMARY_RULE, simulate_buffer  # noqa: E402
from qt.trials import count_unique, deflated_sharpe, load_unique, log_trial  # noqa: E402
from qt.universe import FILTER_EXPRS, sellable_mask, tradable_mask  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")

H = MAIN_HORIZON
LADDER = ("equal", "icir", "max_icir", "linear", "lgbm")
METHOD_NAMES = {"equal": "等权", "icir": "ICIR 加权", "max_icir": "最大化 ICIR", "linear": "线性回归", "lgbm": "LightGBM"}


def display(step: str) -> str:
    """equal_smooth3 → 等权 + 平滑 3 天。"""
    method, _, k = step.partition("_smooth")
    return METHOD_NAMES.get(method, method) + (f" + 平滑 {k} 天" if k else "")
UPGRADE_T = 2.0
UPGRADE_LAG = 5
VETO_SHARE = 0.2
VETO_COMPARISONS = (0.1, 0.0)
SMOOTHING = (3, 5)
INNER_SPLIT = "2017-01-01"  # LightGBM：此前训练，此后早停
TOPK_N_DROP = 3
CACHE_DIR = ROOT / "data" / "cache"


def _data_version() -> str:
    f = DATA_DIR / ".qt_data_version"
    return f.read_text().strip() if f.exists() else "unknown"


def _cached(key: dict, build, refresh: bool):
    """按 key 的哈希缓存到 data/cache/。key 里放会影响结果的全部参数。"""
    name = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE_DIR / f"{key['what']}_{name}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    out = build()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


# 并行中性化：子进程通过 fork 继承这些数据
_CTX: dict = {}


def _neutral_one(name: str) -> pd.Series:
    c = _CTX
    return neutral_score(c["df"][name], c["industry"], c["log_mcap"]) * c["directions"][name]


def load_segment(seg: str, market: str, selected: pd.DataFrame, jobs: int) -> dict:
    """取一个样本段的数据：中性化后的因子分数、1 天和 5 天远期收益、能买能卖。"""
    exprs = {f.name: f.expr for name in FACTOR_SETS for f in FACTOR_SETS[name]() if f.name in selected.index}
    exprs |= {f"ret{h}": forward_return_expr(h) for h in (1, H)} | FILTER_EXPRS | EXPOSURE_EXPRS
    start, end = segment(seg)
    print(f"读取 {market} {seg} 段 {start} ~ {end} ...")
    df = drop_embargo(load_features(exprs, market, start, end), MAX_FORWARD_DAYS)
    industry, log_mcap, _, _ = exposures(df)
    names = list(selected.index)
    _CTX.update(df=df, industry=industry, log_mcap=log_mcap, directions=selected["direction"].to_dict())
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as pool:
        scores = pd.DataFrame(dict(zip(names, pool.map(_neutral_one, names))))
    _CTX.clear()
    return {"scores": scores, "ret1": df["ret1"], "ret5": df[f"ret{H}"],
            "can_buy": tradable_mask(df), "can_sell": sellable_mask(df)}


def load_baseline_segment(seg: str, market: str) -> dict:
    """对照基准：Alpha158 的 158 个因子，每天横截面去极值、标准化，不做中性化。"""
    feats = {f.name: f.expr for f in alpha158()}
    exprs = feats | {f"ret{h}": forward_return_expr(h) for h in (1, H)} | FILTER_EXPRS
    start, end = segment(seg)
    print(f"读取 Alpha158 {market} {seg} 段 ...")
    df = drop_embargo(load_features(exprs, market, start, end), MAX_FORWARD_DAYS)
    x = standardize(df[list(feats)]).astype(np.float32)
    return {"x": x, "ret1": df["ret1"], "ret5": df[f"ret{H}"], "can_buy": tradable_mask(df), "can_sell": sellable_mask(df)}


def inner_split(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """训练段内部切分：INNER_SPLIT 之前训练、之后早停，中间去掉 MAX_FORWARD_DAYS 个交易日，标签不重叠。"""
    dates = index.get_level_values("datetime")
    cut = pd.Timestamp(INNER_SPLIT)
    before = pd.DatetimeIndex(dates[dates < cut].unique()).sort_values()
    fit_end = before[-MAX_FORWARD_DAYS - 1]
    return np.asarray(dates <= fit_end), np.asarray(dates >= cut)


def train_lgbm(x: pd.DataFrame, ret5: pd.Series, can_buy: pd.Series):
    label = combine.rank_label(ret5.where(can_buy))
    fit, stop = inner_split(x.index)
    return combine.fit_lgbm(x[fit], label[fit], x[stop], label[stop])


def estimate_weights(train: dict, scoring: list[str]) -> tuple[dict[str, pd.Series], object, int, pd.DataFrame]:
    s = train["scores"][scoring].where(train["can_buy"], axis=0)
    ic = pd.DataFrame({n: ic_series(s[n], train["ret5"], "spearman") for n in scoring})
    label = combine.rank_label(train["ret5"].where(train["can_buy"]))
    weights = {
        "equal": combine.equal_weights(scoring),
        "icir": combine.icir_weights(ic),
        "max_icir": combine.max_icir_weights(ic),
        "linear": combine.linear_weights(s, label),
    }
    model, best_iter = train_lgbm(s, train["ret5"], train["can_buy"])
    return weights, model, best_iter, ic


def method_score(method: str, scores: pd.DataFrame, weights: dict, model) -> pd.Series:
    if method == "lgbm":
        return combine.predict_lgbm(model, scores)
    return combine.composite(scores, weights[method])


def evaluate(score: pd.Series, seg: dict, buy_ok: pd.Series) -> tuple[pd.DataFrame, dict]:
    sim = simulate_buffer(score, seg["ret1"], buy_ok, seg["can_sell"], PRIMARY_RULE)
    ic = ic_stats(ic_series(score.where(seg["can_buy"]), seg["ret5"], "spearman"), H)
    net = sim["net"]
    metrics = {"rank_ic": ic["mean"], "icir": ic["icir"], "t_nw": ic["t_nw"], **simulation_summary(sim, PRIMARY_RULE),
               "sharpe_ann": net.mean() / net.std() * np.sqrt(TRADING_DAYS), "n_days": len(sim)}
    return sim, metrics


def compare(challenger: pd.DataFrame, incumbent: pd.DataFrame) -> dict[str, float]:
    diff = challenger["net"] - incumbent["net"]
    return {"diff_ann": diff.mean() * TRADING_DAYS, "diff_t": newey_west_t(diff, lag=UPGRADE_LAG)}


def wins(c: dict) -> bool:
    return bool(c["diff_ann"] > 0 and c["diff_t"] > UPGRADE_T)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--refresh", action="store_true", help="不用缓存，重新取数")
    args = ap.parse_args()

    sel_path = ROOT / "reports" / "screening" / f"{args.market}_train_neutral" / "selected.csv"
    selected = pd.read_csv(sel_path, index_col="factor")
    scoring = list(selected.index[selected["category"] == "打分因子"])
    veto = list(selected.index[selected["category"] == "排雷因子"])
    print(f"打分因子 {len(scoring)} 个，排雷因子 {len(veto)} 个")

    version = _data_version()
    base_key = {"market": args.market, "factors": sorted(selected.index), "directions": selected["direction"].to_dict(),
                "data_version": version}
    segs = {s: _cached(base_key | {"what": f"scores_{s}"}, lambda s=s: load_segment(s, args.market, selected, args.jobs),
                       args.refresh) for s in ("train", "valid")}
    train, valid = segs["train"], segs["valid"]

    # 3. 训练段估权重
    print("训练段估计权重 ...")
    weights, model, best_iter, ic_train = estimate_weights(train, scoring)
    weight_table = pd.DataFrame({METHOD_NAMES[m]: w for m, w in weights.items()})
    importance = pd.Series(model.feature_importance("gain"), index=scoring)
    weight_table["LightGBM 重要性"] = importance / importance.sum()

    # 4. 验证段逐级比较
    veto_score = combine.composite(valid["scores"][veto], combine.equal_weights(veto))
    buy_ok = {share: valid["can_buy"] & ~combine.veto_block(veto_score, share) for share in (VETO_SHARE, *VETO_COMPARISONS)}
    params = {"market": args.market, "segment": "valid", "rule": PRIMARY_RULE.label, "benchmark": "buyable",
              "scoring": scoring, "veto": veto, "label": f"rank_ret{H}", "data_version": version}

    rows, sims, scores = [], {}, {}
    incumbent = None
    for m in LADDER:
        scores[m] = method_score(m, valid["scores"][scoring], weights, model)
        train_ic = ic_stats(ic_series(method_score(m, train["scores"][scoring], weights, model).where(train["can_buy"]),
                                      train["ret5"], "spearman"), H)["mean"]
        sims[m], metrics = evaluate(scores[m], valid, buy_ok[VETO_SHARE])
        log_trial("combination", params | {"method": m, "smoothing": 1, "veto_share": VETO_SHARE}, metrics)
        row = {"step": m, "train_rank_ic": train_ic, **metrics}
        if incumbent is None:
            row |= {"vs": None, "diff_ann": np.nan, "diff_t": np.nan, "adopted": True}
            incumbent = m
        else:
            c = compare(sims[m], sims[incumbent])
            row |= {"vs": incumbent, **c, "adopted": wins(c)}
            if wins(c):
                incumbent = m
        rows.append(row)
        print(f"{display(m)}：扣成本后 {metrics['net_ann']:.2%}，{'采用' if row['adopted'] else '不升级'}")
    chosen_method = incumbent

    # 5. 信号平滑
    chosen, chosen_score, chosen_sim = chosen_method, scores[chosen_method], sims[chosen_method]
    smooth_rows = []
    for k in SMOOTHING:
        s = combine.smooth(scores[chosen_method], k)
        sim, metrics = evaluate(s, valid, buy_ok[VETO_SHARE])
        log_trial("combination", params | {"method": chosen_method, "smoothing": k, "veto_share": VETO_SHARE}, metrics)
        c = compare(sim, chosen_sim)
        smooth_rows.append({"step": f"{chosen_method}_smooth{k}", "train_rank_ic": np.nan, **metrics,
                            "vs": chosen, **c, "adopted": wins(c)})
        if wins(c):
            chosen, chosen_score, chosen_sim = f"{chosen_method}_smooth{k}", s, sim
    smoothing = int(chosen.split("smooth")[1]) if "smooth" in chosen else 1

    # 排雷对照
    veto_rows = []
    for share in VETO_COMPARISONS:
        sim, metrics = evaluate(chosen_score, valid, buy_ok[share])
        log_trial("combination", params | {"method": chosen_method, "smoothing": smoothing, "veto_share": share}, metrics)
        veto_rows.append({"veto_share": share, **metrics, **compare(sim, chosen_sim)})
    _, chosen_metrics = evaluate(chosen_score, valid, buy_ok[VETO_SHARE])
    veto_rows.insert(0, {"veto_share": VETO_SHARE, **chosen_metrics, "diff_ann": np.nan, "diff_t": np.nan})

    # 6. 对照基准
    print("对照基准 Alpha158 + LightGBM ...")
    bkey = {"market": args.market, "data_version": version}
    btrain = _cached(bkey | {"what": "alpha158_train"}, lambda: load_baseline_segment("train", args.market), args.refresh)
    bvalid = _cached(bkey | {"what": "alpha158_valid"}, lambda: load_baseline_segment("valid", args.market), args.refresh)
    bmodel, b_iter = train_lgbm(btrain["x"], btrain["ret5"], btrain["can_buy"])
    del btrain
    base_score = combine.predict_lgbm(bmodel, bvalid["x"])
    base_sim, base_metrics = evaluate(base_score, bvalid, bvalid["can_buy"])
    log_trial("baseline", {k: v for k, v in params.items() if k not in ("scoring", "veto")} |
              {"model": "alpha158_lgbm", "features": "cs_standardize", "veto_share": 0.0}, base_metrics)

    # 7. Qlib 正式回测
    print("Qlib 正式回测 ...")
    runs = {
        "选定方案": dict(score=chosen_score, buy_allowed=buy_ok[VETO_SHARE]),
        "选定方案，成本翻倍": dict(score=chosen_score, buy_allowed=buy_ok[VETO_SHARE], cost_multiplier=2.0),
        "选定方案，开盘价成交": dict(score=chosen_score, buy_allowed=buy_ok[VETO_SHARE], deal_price="open"),
        f"选定方案，TopkDropout（每天最多换 {TOPK_N_DROP} 只，不排雷）": dict(score=chosen_score, strategy="topk_dropout", n_drop=TOPK_N_DROP),
        "对照基准 Alpha158 + LightGBM": dict(score=base_score, buy_allowed=bvalid["can_buy"]),
    }
    reports = {name: run_backtest(**kw) for name, kw in runs.items()}
    bt = pd.DataFrame({name: report_metrics(r) for name, r in reports.items()}).T

    # 8. 通缩夏普比率
    n_train = train["ret1"].index.get_level_values("datetime").nunique()
    dsr = _dsr(chosen_sim["net"], n_train)

    out_dir = ROOT / "reports" / "combination" / f"{args.market}_valid"
    out_dir.mkdir(parents=True, exist_ok=True)
    ladder = pd.DataFrame(rows + smooth_rows).set_index("step")
    ladder.to_csv(out_dir / "ladder.csv", float_format="%.6f")
    weight_table.to_csv(out_dir / "weights.csv", float_format="%.6f")
    bt.to_csv(out_dir / "backtest.csv", float_format="%.6f")
    chosen_cfg = {"method": chosen_method, "smoothing": smoothing, "veto_share": VETO_SHARE, "rule": PRIMARY_RULE.label,
                  "scoring": scoring, "veto": veto, "weights": weights.get(chosen_method, pd.Series(dtype=float)).to_dict(),
                  "lgbm_best_iteration": best_iter}
    (out_dir / "chosen.json").write_text(json.dumps(chosen_cfg, ensure_ascii=False, indent=2))
    _plots(sims, chosen_sim, base_sim, reports, out_dir)
    _write_report(out_dir, args, scoring, veto, weight_table, best_iter, b_iter, ladder, chosen, veto_rows,
                  base_metrics, chosen_metrics, bt, reports, chosen_sim, dsr)
    print(bt.round(4).to_string())
    print(f"选定：{chosen}；DSR {dsr['dsr']:.3f}；试验共 {dsr['n_trials']} 个")


def _dsr(net: pd.Series, n_train: int) -> dict:
    """试验次数和夏普方差取自试验日志。旧的单因子试验没记 sharpe_ann，用 t 值换算（都在训练段上）。"""
    sharpes = []
    for r in load_unique():
        m = r["metrics"]
        if "sharpe_ann" in m:
            sharpes.append(m["sharpe_ann"])
        elif "net_t" in m and m["net_t"] is not None:
            sharpes.append(m["net_t"] / np.sqrt(n_train) * np.sqrt(TRADING_DAYS))
    sharpes = np.array([s for s in sharpes if np.isfinite(s)])
    out = deflated_sharpe(net.to_numpy(), len(sharpes), sharpes.var(ddof=1) / TRADING_DAYS)
    return out | {"n_trials": len(sharpes), "sharpe_ann": out["sharpe"] * np.sqrt(TRADING_DAYS),
                  "threshold_ann": out["sharpe_threshold"] * np.sqrt(TRADING_DAYS), "sharpe_std_ann": sharpes.std(ddof=1)}


def _plots(sims, chosen_sim, base_sim, reports, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for m, sim in sims.items():
        sim["net"].cumsum().plot(ax=ax, label=m)
    chosen_sim["net"].cumsum().plot(ax=ax, label="chosen", color="black", lw=2, ls="--")
    base_sim["net"].cumsum().plot(ax=ax, label="Alpha158+LGBM", color="grey", lw=1.5)
    ax.set_title("Validation: cumulative net excess vs buyable universe (buffer 30/90, veto 20%)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ladder.png", dpi=110)
    plt.close(fig)

    labels = {"选定方案": "chosen", "选定方案，成本翻倍": "chosen, 2x cost", "选定方案，开盘价成交": "chosen, open price",
              "对照基准 Alpha158 + LightGBM": "Alpha158+LGBM"}
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, rep in reports.items():
        excess = rep["return"] - rep["cost"] - rep["bench"]
        excess.cumsum().plot(ax=ax, label=labels.get(name, "chosen, TopkDropout"))
    ax.set_title("Qlib backtest (validation): cumulative net excess vs CSI 300 index")
    ax.axhline(0, color="grey", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "backtest.png", dpi=110)
    plt.close(fig)


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def _write_report(out_dir, args, scoring, veto, weight_table, best_iter, b_iter, ladder, chosen, veto_rows,
                  base_metrics, chosen_metrics, bt, reports, chosen_sim, dsr) -> None:
    start, end = chosen_sim.index.min().date(), chosen_sim.index.max().date()

    lad = pd.DataFrame({
        "方案": [display(s) for s in ladder.index],
        "训练段 RankIC": ladder["train_rank_ic"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}"),
        "验证段 RankIC": ladder["rank_ic"].map("{:.4f}".format),
        "验证段 t 值": ladder["t_nw"].map("{:.2f}".format),
        "年化超额": ladder["excess_ann"].map(_pct),
        "成本": ladder["cost_ann"].map(_pct),
        "扣成本后": ladder["net_ann"].map(_pct),
        "日均换手": ladder["turnover"].map(_pct),
        "比较对象": ladder["vs"].map(lambda v: display(v) if isinstance(v, str) else ""),
        "差距（年化）": ladder["diff_ann"].map(_pct),
        "差距 t 值": ladder["diff_t"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
        "结果": ladder["adopted"].map({True: "采用", False: "不升级"}),
    })
    lad.iloc[0, -1] = "起点"

    vt = pd.DataFrame(veto_rows)
    vt = pd.DataFrame({
        "排雷比例": vt["veto_share"].map(lambda v: f"{v:.0%}" if v else "不排雷"),
        "验证段 RankIC": vt["rank_ic"].map("{:.4f}".format),
        "年化超额": vt["excess_ann"].map(_pct), "成本": vt["cost_ann"].map(_pct), "扣成本后": vt["net_ann"].map(_pct),
        "日均换手": vt["turnover"].map(_pct), "和 20% 的差距": vt["diff_ann"].map(_pct),
        "差距 t 值": vt["diff_t"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}"),
    })

    bt_show = pd.DataFrame({
        "年化收益": bt["return_ann"].map(_pct), "沪深300": bt["bench_ann"].map(_pct), "年化超额": bt["excess_ann"].map(_pct),
        "信息比率": bt["info_ratio"].map("{:.2f}".format), "超额最大回撤": bt["excess_max_drawdown"].map(_pct),
        "成本": bt["cost_ann"].map(_pct), "日均换手（双边）": bt["turnover"].map(_pct),
    }, index=bt.index)

    main_bt, base_bt = bt.loc["选定方案"], bt.loc["对照基准 Alpha158 + LightGBM"]
    criteria = {
        "扣成本后年化超额 > 0": main_bt["excess_ann"] > 0,
        "信息比率 ≥ 0.5": main_bt["info_ratio"] >= 0.5,
        "成本翻倍后年化超额 > 0": bt.loc["选定方案，成本翻倍", "excess_ann"] > 0,
        "赢过对照基准（年化超额更高）": main_bt["excess_ann"] > base_bt["excess_ann"],
    }
    rep = reports["选定方案"]
    sim_gross = chosen_sim["portfolio"].mean() * TRADING_DAYS
    qlib_gross = rep["return"].mean() * TRADING_DAYS

    lines = [
        "# 因子合成与验证段回测",
        "",
        f"- 样本：{args.market} 历史成分；权重只用训练段（2010–2018）估计，比较在验证段 {start} ~ {end}（段尾去掉 {MAX_FORWARD_DAYS} 个交易日）",
        f"- 打分因子 {len(scoring)} 个：{', '.join(scoring)}",
        f"- 排雷因子 {len(veto)} 个：{', '.join(veto)}；等权合成排雷分，每天最低 {VETO_SHARE:.0%} 不能新买，已持有的不受影响",
        f"- 交易规则：每日决策，持仓 {PRIMARY_RULE.n_hold}，缓冲 {PRIMARY_RULE.label}；T 日收盘出信号，T+1 收盘成交",
        f"- 升级标准：和当前采用的方案比，每日扣成本超额之差年化 > 0 且 Newey-West t 值（滞后 {UPGRADE_LAG} 阶）> {UPGRADE_T:g}",
        "",
        f"**选定方案：{display(chosen)}**",
        "",
        "## 逐级比较（近似模拟，超额相对第二天能买进的股票）",
        "",
        lad.to_markdown(index=False),
        "",
        "![ladder](ladder.png)",
        "",
        f"LightGBM 在 2017–2018 早停，用了 {best_iter} 轮。训练段 RankIC 是样本内的，LightGBM 的偏高属正常。",
        "",
        "## 权重",
        "",
        "各方法的权重按绝对值之和为 1 归一（合成后会再标准化，只有相对大小有意义）。最后一列是 LightGBM 的特征重要性（按增益）。",
        "",
        weight_table.to_markdown(floatfmt=".3f"),
        "",
        "## 排雷比例对照",
        "",
        vt.to_markdown(index=False),
        "",
        "## 对照基准：Alpha158 + LightGBM",
        "",
        f"158 个因子每天横截面去极值、标准化（不做中性化），同样的标签、切分和参数（早停用了 {b_iter} 轮），同样按缓冲 {PRIMARY_RULE.label} 交易，不排雷。",
        f"验证段 RankIC {base_metrics['rank_ic']:.4f}，扣成本后年化超额 {base_metrics['net_ann']:.2%}（选定方案 {chosen_metrics['net_ann']:.2%}），日均换手 {base_metrics['turnover']:.2%}。",
        "",
        "## Qlib 正式回测（相对沪深300 指数）",
        "",
        f"账户 {ACCOUNT / 1e8:g} 亿元，按 1 手取整，权重随价格漂移，成本按 `qt/costs.py`（最低 5 元），涨跌停 9.5%。收益按日累加计算年化。",
        "",
        bt_show.to_markdown(),
        "",
        "![backtest](backtest.png)",
        "",
        f"和近似模拟对照：选定方案扣成本前的年化收益，近似模拟 {sim_gross:.2%}，Qlib {qlib_gross:.2%}。",
        "相对沪深300 指数的超额，比相对能买股票的超额多了等权和市值加权的差、以及买不进涨停股的损失，两者不能直接比。",
        "",
        "## 通缩夏普比率",
        "",
        f"- 选定方案验证段每日扣成本超额（相对能买进的股票）的年化夏普比率 {dsr['sharpe_ann']:.2f}",
        f"- 试验日志里不同配置共 {dsr['n_trials']} 个，年化夏普比率的标准差 {dsr['sharpe_std_ann']:.2f}；"
        f"全是碰运气时，最好那次的年化夏普比率期望约 {dsr['threshold_ann']:.2f}",
        f"- DSR = {dsr['dsr']:.3f}（真实夏普大于 0 的概率，扣除了多次试验和收益分布的偏度、峰度；常用门槛 0.95）",
        "- 近似：大部分试验是训练段上的单因子检验，和验证段上的合成试验放在一起估计方差",
        "",
        "## 最终测试四条标准在验证段上的预演",
        "",
        "最终测试段还没解锁，这里只是用验证段看一下离标准有多远。",
        "",
        "| 标准 | 验证段 |",
        "|---|---|",
        *[f"| {k} | {'满足' if v else '不满足'} |" for k, v in criteria.items()],
        "",
        "文件：`ladder.csv`（逐级比较）、`weights.csv`（权重）、`backtest.csv`（正式回测）、`chosen.json`（选定方案的配置，最终测试用）。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
