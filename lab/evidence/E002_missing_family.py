"""诊断（便宜证据，不计试验）：某一组分数缺失的股票，在打分顶端的占比和之后的表现。"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from qt.lab import cards as cd, procedure as pc, evidence as ev
from qt.lab.store import load_dev
from qt.composite import category_scores
from run_plan3c_train import POOL_CACHE, latest_version
pd.set_option("display.width", 250)
proc = cd.load_baseline("lab/baseline.yaml")
dev = load_dev(latest_version(POOL_CACHE), proc.factors())
group_of = {f: g for g, fs in proc.families for f in fs}
cats = category_scores({f: dev["factors"][f] for f in group_of}, group_of)
score = pc.composite(dev, proc)
buy = dev["can_buy"].reindex(index=score.index, columns=score.columns).fillna(False)
has_score = score.notna() & buy
flags = {f"缺{g}": cats[g].isna() & has_score for g in cats}
print("能买且有总分的股票里，各组缺失的比例（按年）：")
yr = lambda f: (f.sum(axis=1).groupby(f.index.year).sum() / has_score.sum(axis=1).groupby(has_score.index.year).sum())
print(pd.DataFrame({k: yr(v) for k, v in flags.items()}).round(3).T.to_string())
print()
print(ev.flag_split(score, dev["close"], buy, flags, top=15).round(4).to_string())
print()
print(ev.flag_split(score, dev["close"], buy, flags, top=150).round(4).to_string())
