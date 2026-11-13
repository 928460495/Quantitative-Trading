"""诊断（便宜证据，不计试验）：基线账户里平仓亏损超过 40% 的持仓，买入那天的样子。"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from qt.lab import cards as cd, scorecard as sc, procedure as pc
from qt.lab.store import load_dev
from qt.composite import category_scores
from run_plan3c_train import POOL_CACHE, latest_version
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
proc = cd.load_baseline("lab/baseline.yaml")
dev = load_dev(latest_version(POOL_CACHE), proc.factors())
res = sc.run_phases(dev, proc)
group_of = {f: g for g, fs in proc.families for f in fs}
cats = category_scores({f: dev["factors"][f] for f in group_of}, group_of)
score = pc.composite(dev, proc)
r250 = dev["close"] / dev["close"].shift(250) - 1
rows = []
for k, r in res.items():
    pos = sc.closed_positions(r)
    pos["offset"] = k
    rows.append(pos)
pos = pd.concat(rows)
pos["sig"] = [dev["close"].index[dev["close"].index.get_loc(d) - 1] for d in pos["buy_date"]]   # 信号日
def at(frame, d, c):
    try: return frame.at[d, c]
    except KeyError: return np.nan
for name in ("deduct_loss", "small30", "high_vol", "neg_equity", "loss_small"):
    pos[name] = [bool(at(dev["flags"][name], d, c)) for d, c in zip(pos["sig"], pos["instrument"])]
for g in cats:
    pos["z_" + g] = [at(cats[g], d, c) for d, c in zip(pos["sig"], pos["instrument"])]
pos["rank"] = [score.loc[d].rank(ascending=False).get(c, np.nan) for d, c in zip(pos["sig"], pos["instrument"])]
pos["r250"] = [at(r250, d, c) for d, c in zip(pos["sig"], pos["instrument"])]
pos["st_later"] = [bool(dev["flags"]["is_st"].loc[b:s, c].any()) for b, s, c in zip(pos["buy_date"], pos["sell_date"], pos["instrument"])]
pos["era3"] = pos["sell_date"] >= "2019-01-01"
big = pos[pos["ret"] <= -0.40]
allp = pos
print("全部平仓笔数", len(allp), " 大亏笔数", len(big), " 其中阶段三", int(big["era3"].sum()))
def prof(df):
    cols = ["deduct_loss", "small30", "high_vol", "neg_equity", "st_later"] + [c for c in df if c.startswith("z_")] + ["rank", "r250", "ret"]
    return df[cols].astype(float).mean()
print(pd.DataFrame({"大亏(阶段三)": prof(big[big.era3]), "大亏(阶段一二)": prof(big[~big.era3]),
                    "全部(阶段三)": prof(allp[allp.era3]), "全部(阶段一二)": prof(allp[~allp.era3])}).round(3).to_string())
print(big[big.era3].drop_duplicates("instrument")[["instrument", "buy_date", "sell_date", "ret", "rank", "r250", "deduct_loss", "small30", "st_later"] + [c for c in big if c.startswith("z_")]].round(2).to_string())
