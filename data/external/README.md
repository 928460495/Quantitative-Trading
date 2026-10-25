# 外部数据

行业和股本数据。这里的网络（云端环境）访问不了数据源，需要在本地采集后提交到仓库。

## 怎么采集

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-collect.txt
python scripts/download_data.py          # 已经有 Qlib 数据就跳过；采集脚本要从这里读股票名单
python scripts/collect_external.py       # 沪深300 历史上的全部成分股，约 20–30 分钟
```

中途断了直接重跑：已下载的股票会跳过，失败的会补。跑完检查屏幕上打印的两行：

- "按前两位归类得到 N 个一级行业"：申万 2021 版应该是 31 个。不是的话先别提交，告诉我实际的代码长度分布
- "失败 N 只"：应该是 0。不是的话再跑一次

然后提交这三个文件：

```bash
git add data/external/sw_industry_hist.csv data/external/share_changes.csv data/external/collect_meta.json
```

## 文件

| 文件 | 来源 | 列 |
|---|---|---|
| `sw_industry_hist.csv` | 申万宏源研究，AkShare `stock_industry_clf_hist_sw()` | `symbol`（6 位代码）、`start_date`（计入日期）、`industry_code`（申万 2021 行业代码，前两位是一级行业） |
| `share_changes.csv` | 巨潮资讯，AkShare `stock_share_change_cninfo()` | `symbol`、`declare_date`（公告日期）、`change_date`（变动日期）、`effective_date`（两者中较晚的一个）、`total_shares`（总股本）、`float_a_shares`（流通 A 股，取"人民币普通股"，缺失时取"已流通股份"） |
| `collect_meta.json` | 采集脚本 | 采集时间、AkShare 版本、股票数、失败清单 |

股本单位沿用巨潮的原始口径。中性化用的是对数流通市值，单位只差一个常数，不影响结果。

## 怎么用

```bash
python scripts/run_single_factor.py --neutralize
```

报告写到 `reports/single_factor/csi300_train_neutral/`，汇总表最后有和未中性化结果的对比。
