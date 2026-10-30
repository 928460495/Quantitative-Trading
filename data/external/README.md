# 外部数据

行业和股本数据。这里的网络（云端环境）访问不了数据源，需要在本地采集后提交到仓库。

## 怎么采集

采集只需要 pandas 和 AkShare，不用装 Qlib。Python 3.10 及以上的正式版都可以（不要用 3.12 的 alpha / beta 测试版）。

```bash
python --version                          # 确认是正式版，例如 Python 3.12.7
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-collect.txt
python scripts/download_data.py           # 已经有 Qlib 数据就跳过；采集脚本要从这里读股票名单
python scripts/collect_external.py        # 沪深300 历史上的全部成分股，约 20–30 分钟
```

用 conda 的话，把前两行换成 `conda create -n qt python=3.12 -y && conda activate qt`。

中途断了直接重跑：已下载的股票会跳过，失败的会补。跑完检查屏幕上打印的两行：

- "按前两位归类得到 N 个一级行业"：申万 2021 版应该是 31 个。不是的话先别提交，告诉我实际的代码长度分布
- "失败 N 只"：应该是 0。不是的话再跑一次

### 行业表下载失败（SSL 证书错误）时

申万网站的证书在一些电脑上会校验失败（`CERTIFICATE_VERIFY_FAILED`）。这时用浏览器打开下面的地址手动下载，再让脚本读本地文件：

```
https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls
```

```bash
python scripts/collect_external.py --only industry --industry-file ~/Downloads/StockClassifyUse_stock.xls
```

行业表下载失败不会中断股本的采集，股本照常进行。

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

## 全 A 数据（baostock、东方财富、中证指数）

全 A 的日线和财务数据量大（几百 MB），不提交到仓库，打包后上传到本仓库的 GitHub Release，云端再下载。

注意：仓库是公开的，Release 里的文件任何人都能下载。

### 1. 拿到采集脚本

在仓库目录里依次运行：

```bash
git fetch origin claude/full-a-data
```

```bash
git checkout claude/full-a-data
```

### 2. 装依赖

沿用之前采集行业和股本时的环境（比如 `conda activate qt`），再装一次依赖（新加了 baostock、pyarrow、requests）：

```bash
pip install -r requirements-collect.txt
```

### 3. 采集

```bash
python scripts/collect_full_a.py
```

依次做五步：股票列表 → 日线（约 5500 只，1–3 小时）→ 季度财务（约 300 次请求）→ 指数 → 打包。中途断了直接重跑同一条命令，做过的会跳过。

跑完检查屏幕上的输出：

- `daily：{...}` 里 `failed` 应该是 0；不是的话再跑一次（只会补失败的）
- 每年覆盖情况表：`daily` 应该接近 `listed`；财务表（`income` 等）明显少于 `listed` 的年份，把这张表发给 Claude
- 最后一行是生成的文件，例如 `data/full_a_20261027.tar`

### 4. 上传到 GitHub Release

1. 打开 https://github.com/928460495/Quantitative-Trading/releases/new
2. "Choose a tag" 里输入新 tag，例如 `full-a-20261027`（和文件名里的日期一致），选 "Create new tag"
3. 标题随便写，例如"全 A 数据 2026-10-27"
4. 把 `data/full_a_20261027.tar` 拖进附件区，等上传完成（单个文件上限 2 GB）
5. 点 "Publish release"

然后把 tag 告诉 Claude。
