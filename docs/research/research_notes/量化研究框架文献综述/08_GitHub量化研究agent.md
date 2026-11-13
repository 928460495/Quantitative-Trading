# GitHub 上"用 AI agent 做量化策略研究"的开源项目：研究循环实现与可移植设计

说明：本次只读，没有克隆、安装或运行任何仓库代码。本机没有 `gh`，改用 GitHub REST API 和 raw 文件读取（读到的是 2026-11-12 前后的 main 分支）。作者自报的收益数字都标了"作者自报"。按要求，不对 2022 年以后的 A 股因子表现下"哪类因子好"的结论；下文出现的 2022 年后数字只是别人 README 里的自报，且仅用来说明"他们的测试段设在哪"。

## 1. microsoft/RD-Agent（重点）

### Takeaway
RD-Agent 的"假设 → 实验 → 回测 → 反馈 → 下一轮"循环结构清楚、值得学，但它的"是否采纳"由 LLM 按一句话规则主观判断，没有显著性门槛；而且循环里反复看的恰好是配置里的 test 段（默认 2017–2020），等于把测试集当成了搜索用的验证集。整套需要 LLM API，也基本不能不装 Docker/大量依赖，在我们的环境里不适合直接装，适合移植设计。

### Cited Findings

**仓库状态**
- 约 1.5 万 star，MIT 许可，最近一次推送 2026-10-21，open issue 约 247 个 — [GitHub API](https://api.github.com/repos/microsoft/RD-Agent)
- 量化场景叫 RD-Agent(Q)，论文 NeurIPS 2025（arXiv 2505.15155）。README 称"成本低于 10 美元、年化超额收益约为基准因子库的 2 倍、因子数少 70% 以上"（作者自报）— [README](https://github.com/microsoft/RD-Agent/blob/main/README.md)

**循环步骤（代码位置）**
- 循环骨架在 `rdagent/components/workflow/rd_loop.py` 的 `RDLoop`，每一轮按顺序执行：`direct_exp_gen`（提假设并转成实验）→ `coding`（写因子代码，CoSTEER）→ `running`（跑 qlib 回测）→ `feedback`（LLM 总结，出错则直接 decision=False）→ `record`（写入 trace）— [rd_loop.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/workflow/rd_loop.py)
- 入口是 `rdagent/app/qlib_rd_loop/` 下的 `factor.py`、`model.py`、`quant.py`（因子与模型联合）、`factor_from_report.py`（从研报抽因子）；配置在同目录 `conf.py` — [目录](https://github.com/microsoft/RD-Agent/tree/main/rdagent/app/qlib_rd_loop)
- 联合模式（quant）用一个线性 Thompson 抽样的两臂老虎机决定本轮"改因子"还是"改模型"；上下文特征是 IC、ICIR、RankIC、年化超额、IR、回撤、夏普等 8 项 — [bandit.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/proposal/bandit.py)

**Hypothesis 对象的字段**
- `rdagent/core/proposal.py` 里 `Hypothesis` 有六个字符串字段：`hypothesis`、`reason`、`concise_reason`、`concise_observation`、`concise_justification`、`concise_knowledge`。没有任何数值字段（没有预期方向、预期 IC、检验日期、证伪条件）— [proposal.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/core/proposal.py)
- 生成假设的提示词（`rdagent/scenarios/qlib/prompts.yaml` 的 `factor_hypothesis_output_format`）只要求输出 JSON：`hypothesis`（2–3 句）和 `reason`（2–3 句）。规则里写"先简单后复杂""连续多轮没超过 SOTA 就换方向""超过 SOTA 的因子已入库，别重复实现" — [prompts.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/prompts.yaml)
- 提案代码里有一条硬编码的"RAG"提示：历史不足 15 轮时让它"先试最简单最快的因子"，之后改成"试能拿高 IC 的因子（如机器学习因子）" — [factor_proposal.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/proposal/factor_proposal.py)

**"比 SOTA 好"怎样判定**
- `rdagent/scenarios/qlib/developer/feedback.py`：只把三项指标交给 LLM：`IC`、`1day.excess_return_with_cost.annualized_return`、`1day.excess_return_with_cost.max_drawdown`，并排给出"当前 vs SOTA"— [feedback.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/developer/feedback.py)
- 判定是 LLM 输出 JSON 里的 `Replace Best Result`（yes/no）。提示词原话要点：任何小幅改进都应考虑采纳；只要年化收益提高就建议替换；其他指标小幅变差可以接受。所以没有固定阈值，没有显著性检验，没有多重检验校正，是主观判断 — 同上 `prompts.yaml` 的 `factor_feedback_generation`
- 反馈 JSON 字段：`Observations`、`Feedback for Hypothesis`、`New Hypothesis`、`Reasoning`、`Replace Best Result`；对应类 `HypothesisFeedback`（另含 `decision`、`acceptable`、`exception`）— proposal.py

**失败试验如何留痕、喂回下一轮（Trace）**
- `Trace` 的核心是 `hist`：按时间排列的 `(Experiment, ExperimentFeedback)` 列表，加 `dag_parent`（每个节点的父节点，形成树/DAG）和 `idx2loop_id`。`get_sota_hypothesis_and_experiment()` 倒着找第一个 `decision=True` 的节点 — proposal.py
- 失败（代码出错、因子为空、回测超时）会作为 `decision=False` 的反馈入 `hist`，下一轮提示词里通过 `hypothesis_and_feedback` 模板把历史假设和反馈整段塞回去。我没有读到对历史做压缩或检索的逻辑，只是整段拼接（读到了 factor_proposal.py 的 `prepare_context`，未细读模板内容）
- 采纳之后：`exp.based_experiments` = 空基线 + 历史上所有 decision=True 的因子实验，下一轮的新因子与这些 SOTA 因子合并后再回测 — factor_proposal.py

**因子去重**
- 提案阶段：只按因子名字符串精确匹配，重名的丢弃 — factor_proposal.py
- 运行阶段：`factor_runner.py` 的 `deduplicate_new_factors`：新因子与 SOTA 因子逐对算"逐日截面相关系数的时间均值"，与任一 SOTA 因子相关 ≥0.99 就删掉；全被删则抛 `FactorEmptyError`，提示"换方向"— [factor_runner.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/developer/factor_runner.py)
- 阈值 0.99 很松，只能去掉几乎一模一样的因子，去不掉"换皮因子"

**CoSTEER 知识库**
- 位置 `rdagent/components/coder/CoSTEER/knowledge_management.py`。它记的是"写代码"这一层的知识，不是"研究结论"：成功任务的实现与反馈、失败任务的历史尝试。v1 的检索：同一任务已成功则直接复用；失败次数超过 `fail_task_trial_limit` 就标为"放弃"；否则按任务描述的 embedding 相似度取最相似的若干成功案例，外加最近几次失败轨迹 — [knowledge_management.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/coder/CoSTEER/knowledge_management.py)。v2（默认推荐）我没有细读，是缺口
- 因子实现的自动检查（`factor_coder/eva_utils.py`）包含：输出格式、日期索引、行数比例、缺失值、取值全相同比例、是否含 inf，以及与"标准答案"的相关性（>0.99 视为高相关）— [eva_utils.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/components/coder/factor_coder/eva_utils.py)

**训练/验证/测试日期，以及循环里到底看哪一段**
- `conf.py` 默认：train 2008-01-01 至 2014-12-31；valid 2015-01-01 至 2016-12-31；test 2017-01-01 至 2020-08-01 — [conf.py](https://github.com/microsoft/RD-Agent/blob/main/rdagent/app/qlib_rd_loop/conf.py)
- qlib 配置模板 `factor_template/conf_baseline.yaml` 里，回测 `backtest.start_time` 取 `test_start`，`SigAnaRecord`、`PortAnaRecord` 默认汇报 dataset 的 test 段；`read_exp_res.py` 把这些导出成 `exp.result`，`feedback.py` 读的就是它 — [conf_baseline.yaml](https://github.com/microsoft/RD-Agent/blob/main/rdagent/scenarios/qlib/experiment/factor_template/conf_baseline.yaml)
- 一个第三方 PR 的描述证实了这一点：Qlib 汇报的是 test 段，runner 把它直接映射到配置的测试期，迭代反馈和"是否替换"都用它；PR 提议循环改用验证段，测试段只在循环结束后跑一次 — [RD-Agent PR #1442](https://github.com/microsoft/RD-Agent/pull/1442)（截至读取时仍为 open，未合并）
- 同类质疑出现在姊妹项目 QuantaAlpha 的 issue 里，有人说"RD-Agent 也是这样，直接拿测试集收益筛因子池" — [QuantaAlpha issue #17](https://github.com/QuantaAlpha/QuantaAlpha/issues/17)

**运行依赖与费用**
- README：大多数场景需要 Docker，且需要能不加 sudo 运行；需要配置 LLM（OpenAI 兼容接口，实验性支持 DeepSeek）和 embedding 模型 — [README](https://github.com/microsoft/RD-Agent/blob/main/README.md)
- 代码里 qlib 回测环境可选 `docker` 或 `conda`（`workspace.py`，`env_type`）；因子代码执行环境默认是本地 conda（`factor_coder/config.py`）。所以"不用 Docker"在技术上有路，但文档以 Docker 为主，issue 里有人在 podman、MLflow 新版、torch 缺失、回测 3600 秒超时上出问题（#1394、#1464、#1333、#1372）
- 费用量级：README 自报"低于 10 美元"；没有找到独立第三方的费用核算

**其他被指出的问题**
- 开着的 PR 指出 bandit 奖励里回撤符号反了（#1498），以及"ARR 项实际不起作用"（#1479）、"让反馈指标可配置"（#1518）— [PR 列表搜索结果](https://github.com/microsoft/RD-Agent/pulls)。说明奖励/反馈指标这块代码质量不稳，是否合并我没有逐一核实
- 论文给的收益是在 CSI 300 的自报结果，且上面这个默认配置下测试段参与了选择，所以这些收益数字不能当作样本外证据（我的推断，见下）

### Inferences
- "循环里看的是 test 段"这一点会直接造成过拟合：每一轮选择都用测试段信息，试得越多，测试段上的好看结果越多半是挑出来的。我们的 2022 年后封存段不能被这样使用。
- LLM 主观判断采纳 + "小幅改进就采纳" 的规则，等于零门槛，叠加循环次数多，几乎必然累积噪声因子。我们已有的事先登记门槛和 DSR 要求比它严得多。
- RD-Agent 值得照搬的是数据结构（Trace 的 hist + 父子关系、反馈 JSON 的五个字段、SOTA 因子池累积），不值得照搬的是判定方式和日期用法。
- 在我们的环境（单进程 Mac、无 API 预算、已有自己的回测引擎）直接装不现实：它要的是 LLM 自动写代码，而我们是编程助手本人在读写代码；它的 qlib 回测口径与我们"按真实账户记账"的引擎也不一致。

### Gaps
- CoSTEER v2 的检索与记忆细节、`rdagent/app/qlib_rd_loop/prompts.yaml` 和 `hypothesis_and_feedback` 模板的具体内容未读。
- 没找到第三方对 RD-Agent(Q) 收益的独立复现；也没找到费用的独立核算。
- 因子运行时 qlib 的 `test` 段究竟如何被 `FactorBasePropSetting` 覆盖，是我从配置模板推断的，PR #1442 的描述与之一致，但我没有运行验证。

## 2. microsoft/qlib：滚动样本外工具

### Takeaway
qlib 有现成的"滚动切分任务"工具（`RollingGen`）和在线滚动示例，可以生成"训练窗口往前走、每段只测下一段"的任务列表；但它围绕"模型滚动再训练"设计，不是"每年重新选因子"，我们要的是用它的时间切分逻辑，而不是整套流程。

### Cited Findings
- 约 5 万 star，最近推送 2026-11-04 — [GitHub API](https://api.github.com/repos/microsoft/qlib)
- `qlib/workflow/task/gen.py` 的 `RollingGen(step=40, rtype=ROLL_EX|ROLL_SD, trunc_days=...)`：`ROLL_EX` 是起点固定、训练段扩展；`ROLL_SD` 是窗口大小固定、整体滑动；`trunc_days` 用来截掉训练段末尾，避免标签用到未来收益造成泄漏（`trunc_segments`）— [gen.py](https://github.com/microsoft/qlib/blob/HEAD/qlib/workflow/task/gen.py)
- `qlib/contrib/rolling/base.py` 的 `Rolling` 类：把一个任务配置拆成按时间排列的多个任务，离线跑完再拼接成一条完整的样本外预测；文档明确说它只做离线回测，与在线服务（OnlineStrategy）不共用接口 — [base.py](https://github.com/microsoft/qlib/blob/HEAD/qlib/contrib/rolling/base.py)
- 示例：`examples/model_rolling/task_manager_rolling.py`、`examples/online_srv/rolling_online_management.py`、`examples/benchmarks_dynamic/baseline/rolling_benchmark.py`、`examples/rolling_process_data/` — [examples](https://github.com/microsoft/qlib/tree/HEAD/examples)
- 在线管理部分在 `qlib/workflow/online/`（manager、strategy、update）

### Inferences
- 对"每年用过去数据重选因子、往前走一年"，最小做法是自己写一个几十行的年度切分函数（训练 = 起点到 Y−1 年底，评估 = Y 年，并留标签重叠的净空期），参考 `RollingGen` 的 `ROLL_EX` + `trunc_days` 设计；不必依赖 qlib 的任务管理器（它基于 MLflow 和 MongoDB 风格任务池，较重，且我们的回测引擎不是 qlib 的）。
- 这类"滚动选择"能把"选择"本身放进样本外检验：每年的选择规则只用当年之前的数据。

### Gaps
- 没有运行 qlib 的滚动示例，所以不知道在 pandas 3 + pyqlib 0.9.7 上能否直接跑通。

## 3. 其他仓库的评估

### Takeaway
真正带回测评价的研究循环只有少数几个：QuantaAlpha、AlphaAgent、LLM_QUANT_FACTORY（含 AutoAlpha）、alphagen，以及（通用科研类）autoresearch 和 AI-Scientist。TradingAgents、ai-hedge-fund、FinRobot 本质是多角色聊天做交易决策，不是"假设—检验"循环。其中最贴近我们情况的是 LLM_QUANT_FACTORY 的 AutoAlpha 协议和 karpathy/autoresearch 的 `program.md`。

### Cited Findings

**khakhasshi/LLM_QUANT_FACTORY（A 股、纯多、可审计，最相关）**
- 约 70 star，2026-08 仍在更新，许可是 PolyForm Noncommercial（非商用，个人研究够用）；Python 3.12、uv — [仓库](https://github.com/khakhasshi/LLM_QUANT_FACTORY)
- 它的原则：LLM 只提机制假设、生成受控表达式、写结构化意见；数据时点检查、回测、统计检验、组合权重、风险门禁都由确定性代码执行 — [README](https://github.com/khakhasshi/LLM_QUANT_FACTORY/blob/HEAD/README.md)
- `AutoAlpha/program.md` 是给 LLM 的行为规约（与 autoresearch 的 program.md 同类）：研究员每次只提一个可证伪假设，须声明经济机制、因子族、单一变更、预期；评审员独立检查时间语义、重复性、复杂度；执行员只跑已批准的候选。执行顺序固定：数据合约 → purged walk-forward 加 embargo → 对配对成本后收益做 HAC、block bootstrap、DSR、PBO、多重检验 → 稳健性（跨折、跨年、参数邻域、延迟）→ 增量收益 → 换手/费用/容量 → 顺序硬门禁 → holdout 另行审批。IC 只作诊断，不能当准入捷径 — [program.md](https://github.com/khakhasshi/LLM_QUANT_FACTORY/blob/HEAD/AutoAlpha/program.md)
- 反馈政策：研究员只收到"失败门禁名称、预先批准的诊断摘要、重复簇、资源消耗、语义错误"；禁止回传 holdout 序列、逐日盈亏、"离门槛还差多少"这类可反推门槛的信息；同一方向最多 3 次提案，连续 2 次无改善就结束并冷却 2 个战役 — 同上
- `docs/INSTITUTIONAL_RESEARCH_CONTROLS.md`："人工研究污染台账"：每次手动回测记录区间和证据哈希，请求与隐藏测试期重叠就标为污染，同一世代里含被暴露因子的组合无法再进盲测，且换世代 ID 不能清零 — [文档](https://github.com/khakhasshi/LLM_QUANT_FACTORY/blob/HEAD/AutoAlpha/docs/INSTITUTIONAL_RESEARCH_CONTROLS.md)
- `docs/FACTOR_LEAKAGE_AUDIT_20260716.md`（作者自己的审计）：旧评价按"当日收盘信号、当日收盘成交"评价，是执行时点泄漏，改成下一开盘成交后，85 个因子里 9 个夏普保留率低于 50%；人工查看过的区间不能再叫隐藏样本；当前数据没有历史 ST、停牌、涨跌停方向、PIT 行业，只能标 RESEARCH_ONLY — [审计](https://github.com/khakhasshi/LLM_QUANT_FACTORY/blob/HEAD/AutoAlpha/docs/FACTOR_LEAKAGE_AUDIT_20260716.md)
- 因子知识库保存公式与 AST、机制类型、行为簇、同质簇、生命周期、失效标签、边际贡献，用来区分"新机制"和"换皮参数"（README）。去重文档：`AutoAlpha/docs/FACTOR_HOMOGENEITY_CONTROL.md`（未细读）
- 运行依赖：需要 LLM API，工程很重（PostgreSQL 迁移文档、web 界面）；star 少，缺独立验证

**QuantaAlpha/QuantaAlpha（LLM + 进化，A 股/CSI300，约 1.7k star，2026-06 更新，MIT）**
- 流程：研究方向 → 并行规划多个方向 → 每个方向走 5 步（提议/构造/计算/回测/反馈）→ 轨迹进化（变异、交叉）→ 因子库。配置 `configs/experiment.yaml`；进化代码 `quantaalpha/pipeline/evolution/`（controller、mutation、crossover、trajectory）— [仓库](https://github.com/QuantaAlpha/QuantaAlpha)
- `StrategyTrajectory` 数据结构（`trajectory.py`）：trajectory_id、direction_id、round_idx、phase（原始/变异/交叉）、hypothesis、hypothesis_details、factors（名称、表达式、描述、代码）、backtest_metrics、feedback、feedback_details、parent_ids、created_at — 这是比 RD-Agent 更完整的"假设+实验+血缘"记录
- 质量门禁（`experiment.yaml` 的 `quality_gate`）：复杂度检查（符号长度 ≤200、基础特征数 ≤5、自由参数占比 ≤0.5）、冗余检查（对因子库做 AST 子树匹配，阈值 5）、可选的"假设—描述—表达式一致性"检查（LLM，默认关）— [experiment.yaml](https://github.com/QuantaAlpha/QuantaAlpha/blob/HEAD/configs/experiment.yaml)
- README 自报：CSI300、测试期 2022–2025，IC 0.0472、年化 4.68%、最大回撤 11.80%（作者自报）；同一作者的另一份 README 快照（lucaluca-04 的 fork 里）列的是 IC 0.1501、年化 27.75%（作者自报），两者差异很大，我没有找到官方说明原因
- 被指出的问题：issue #17"用测试集 Rank IC 指导进化，是否泄漏或过拟合"（有人把 segments 改成 train 2016–2020、valid 2021、test 2022 起）；issue #27 报告 `runner.py` 里 `isinstance(selector, tuple)` 拒绝了 yaml 里的 list，导致训练和验证都读到全样本（含测试段），声称年化 46.4% 其实真实测试 IC 只有 0.034（用户报告，未经项目方确认）；issue #19 有人用别的模型只复现出 IC 0.03，并质疑结果；issue #31 指出按次日开盘价成交却用当日涨跌幅判断涨停，存在前视 — [issue 列表](https://github.com/QuantaAlpha/QuantaAlpha/issues)
- 依赖：LLM API、Qlib 数据、可选 Docker（`use_docker: false` 可本地）。下一代 QuantaAlpha-claw 声称加入更严格的 walk-forward（README 提到，未读代码）

**RndmVariableQ/AlphaAgent（A 股，约 430 star，2026-07 更新，无许可证）**
- 现在的仓库是"多因子研究框架 + 可选 LLM 挖掘"：Tushare 数据、DSL 因子表达式、FactorZoo、`eval_factor.py` 出 IC/换手/分组报告，挖掘用 AgentScope 智能体（`factor_mining_agentscope.py`）；中证 1000 成分、2015–2026 面板 — [README](https://github.com/RndmVariableQ/AlphaAgent)
- 我没有读到它与论文里"原创性正则化（与已有因子的 AST 相似度惩罚）"的对应代码，不下结论。无许可证，不建议直接复制代码

**karpathy/autoresearch（通用自动科研循环，约 10 万 star，2026-03 最后更新）**
- 整个循环写在一个 `program.md` 里，供编程助手阅读：只能改 `train.py`；`prepare.py`（评价与数据）只读、不准改；固定 5 分钟预算；单一指标 val_bpb；在分支 `autoresearch/<tag>` 上跑；每次实验 git commit → 运行 → 读结果 → 记入未跟踪的 `results.tsv`（列：commit、指标、显存、status=keep/discard/crash、描述）→ 变好就保留提交，不好就 `git reset`；崩溃了小错自己修、大错记 crash 跳过；"简单性准则"：同等结果下越简单越好；"永不停止"— [program.md](https://github.com/karpathy/autoresearch/blob/HEAD/program.md)
- 这正是"编程助手在终端里直接读写代码"的形态，与我们的条件完全吻合；它没有防过拟合机制（单一指标、验证集反复使用），是它的弱点

**SakanaAI/AI-Scientist（约 1.5 万 star，2025-12 最后更新，许可 NOASSERTION）**
- 基于模板的想法生成 → 实验 → 写论文 → 自动评审（LLM 评审）；README 警告会执行 LLM 写的代码，建议沙箱 — [README](https://github.com/SakanaAI/AI-Scientist)。它的评审是 LLM 打分，不是统计检验，对量化的借鉴有限，只有"想法去重（对照已有文献/已有想法）"和"评审员与研究员分离"的思路可取。我没有深读其代码

**ICT-FinD-Lab/alphagen（强化学习找公式因子，约 1.3k star，2026-06 更新）**
- KDD 2023 论文代码，用 MaskablePPO 训练公式生成；目录 `alphagen_llm` 里另附 LLM 迭代生成因子的脚本（`scripts/llm_only.py`）。奖励是因子组合的 IC，不是我们要的"假设—反思"循环；可借鉴点是"因子集合的协同"（新因子看对组合的边际贡献，而非单因子 IC）— [仓库](https://github.com/ICT-FinD-Lab/alphagen)

**TauricResearch/TradingAgents（约 12 万 star）、virattt/ai-hedge-fund（约 6.5 万）、AI4Finance/FinRobot（约 8 千）**
- 三者都是多角色（分析师、研究员、风控、交易员）LLM 对话做交易决策或研究报告，README 都标注"研究/教育用途，非投资建议"。TradingAgents 近期版本加了按日期的时点数据、回测网格和决策日志记忆，更新日志大量是"前视修复"— [TradingAgents README](https://github.com/TauricResearch/TradingAgents)。结论：它们没有"提出可检验假设—统计检验—采纳/拒绝"的循环，对我们价值低，只有"决策日志记忆（事后用实际收益回填结算）"一个小点可参考

**其他 2025–2026 搜索到的小项目（star 很少，未深读，不作推荐）**
- gaosu0715-lgtm/Evolutionary-Alpha-Miner、wzhmtldswz/CosmosAlpha（MCTS/图搜索）、wuyihong994/EvoAlpha、heranliu/AlphaResearchOS、VernonOY/replicalpha（研报 → 因子 → 回测 → 红队 → 可复现性）、chester1uo/Assay（给 LLM 智能体用的因子回测引擎）— [GitHub 搜索](https://github.com/search?q=llm+alpha+mining&type=repositories)。star 个位数到两位数，成熟度和真实性未核实

### Inferences
- 在"真正带回测评价的循环"里，只有 LLM_QUANT_FACTORY 把"研究员看不到什么"（holdout、门槛距离）当成一等设计；其余多数把测试段当反馈用。
- 这些项目都要 LLM API 或 Docker 或重型数据库，没有一个能"装上就用"到我们的 A 股主板 + 真实账户记账体系里；移植设计比安装更合适。

### Gaps
- 没有深读 AlphaAgent 的挖掘代码、AlphaForge（找到的仓库名为加密货币/小项目，与 AlphaForge 论文是否同一项目未确认）和 Alpha-GPT（搜到的 AlphaGPT 仓库是强化学习/符号回归的别项目，与 Alpha-GPT 论文复现是否对应未确认）。
- QuantAgent 搜到的仓库都是无关小项目，没有找到可靠的对应开源项目，跳过。
- 各仓库 star 数、更新时间是 2026-11-12 前后的快照，可能变化。

## 4. 被指出的问题：过拟合、测试集泄漏、不可复现、LLM 前视

### Takeaway
最集中的问题有三类：把测试段当反馈（RD-Agent、QuantaAlpha）、代码 bug 让训练读到测试段（QuantaAlpha #27）、执行时点前视（QuantaAlpha #31、LLM_QUANT_FACTORY 自审）。另有模型权重里的"知识泄漏"，对在 2022 年前后做研究的 LLM 尤其值得警惕。

### Cited Findings
- RD-Agent 循环反馈用 test 段：[PR #1442](https://github.com/microsoft/RD-Agent/pull/1442)；QuantaAlpha 同类：[issue #17](https://github.com/QuantaAlpha/QuantaAlpha/issues/17)
- 训练/验证读到全样本的 bug 报告：[QuantaAlpha #27](https://github.com/QuantaAlpha/QuantaAlpha/issues/27)（用户报告，未经确认）
- 不可复现与质疑：[QuantaAlpha #19](https://github.com/QuantaAlpha/QuantaAlpha/issues/19)（有人称用不同模型只复现出 IC 0.03；语气激烈，仅作信号）
- 执行时点前视：[QuantaAlpha #31](https://github.com/QuantaAlpha/QuantaAlpha/issues/31)；LLM_QUANT_FACTORY 自审 [FACTOR_LEAKAGE_AUDIT](https://github.com/khakhasshi/LLM_QUANT_FACTORY/blob/HEAD/AutoAlpha/docs/FACTOR_LEAKAGE_AUDIT_20260716.md)
- TradingAgents 的 issue #805 指出：即便数据按时点限制，模型权重本身可能已含回测期之后的信息（"模型层面的时间知识泄漏"）— [issue #805](https://github.com/TauricResearch/TradingAgents/issues/805)
- 学术侧（来自搜索摘要，未读全文）：2026 年的 Look-Ahead-Bench（arXiv 2601.13770）、"Summoning the Oracle to Slay It"（arXiv 2605.24564）、"The Alpha Illusion"（arXiv 2605.16895）都讨论了 LLM 参数里的前视偏差，认为仅靠数据管线审计抓不到 — [arXiv 2605.24564](https://arxiv.org/pdf/2605.24564)、[arXiv 2601.13770](https://arxiv.org/pdf/2601.13770)、[arXiv 2605.16895](https://www.alphaxiv.org/abs/2605.16895.md)

### Inferences
- 对我们：编程助手的训练数据很可能覆盖 2022 年以后的市场，它提出假设时可能"凭记忆"偏向后来表现好的主题。所以不能让助手在提假设阶段接触封存段的任何结果，也要在假设中要求写明经济机制和事先预期方向，减少事后凭记忆挑选的空间。这一点是我的推断，不是上述项目的结论。

### Gaps
- 没有读到 RD-Agent 官方对 #1442 的回应；没有找到对 QuantaAlpha #27 的官方确认。

## 5. 结论：最小可行的自进化循环

### Takeaway
不装任何项目。用"autoresearch 的循环形态（编程助手 + 一份 program.md + git + results 日志）"加"AutoAlpha 的反馈与门禁纪律"加"RD-Agent/QuantaAlpha 的假设—实验—血缘数据结构"，在我们已有的因子库、回测引擎、试验日志、DECISIONS.md 上搭一个薄层。

### Cited Findings
（见前四节。此节为综合推断，来源已在各节标出。）

### Inferences

**最小循环（每个假设一圈）与文件**
1. 读状态：`docs/HANDOFF.md`、`reports/trials.jsonl` 中的假设台账、失败原因摘要。
2. 写假设（落盘，运行前）：一个 JSON/YAML 记录，字段取自 RD-Agent `Hypothesis` + AutoAlpha `ResearchProposal` + QuantaAlpha `StrategyTrajectory`：`id`、`parent_id`、`mechanism`（经济机制）、`change`（只改一处）、`expected_direction`、`falsification`（什么结果算证伪）、`family`、`factor_expr_hash`。同时在 `DECISIONS.md` 追加一条（遵守"运行前写规则"）。
3. 去重：对因子库做相关性和表达式结构（AST）检查，阈值比 RD-Agent 的 0.99 严（如横截面时序平均相关高于某事先登记值就算同类，具体数值需我们先定并登记）；同机制族被拒过则要冷却。
4. 实现 + TDD：`qt/` 下写因子和测试。
5. 只在"训练/验证"段评估：采纳规则是事先登记的固定门槛（含成本后的真实账户口径、跨年稳健性、对已有组合的增量），不由 LLM 主观判断；只回传"通过/未过哪道门 + 预设诊断"，不回传离门槛的距离。
6. 记录：无论成败写 `reports/trials.jsonl`（试验次数用于最终 DSR），失败原因分类入台账；采纳的因子进"SOTA 因子池"，带父节点。
7. 复盘：每 N 个假设写一次"认识更新"（哪些机制族被拒、为什么），追加到 `DECISIONS.md`，作为下一轮提假设的输入（这是 RD-Agent 的 trace 反馈 + AutoAlpha 的失败类别摘要）。
8. 滚动检验：每年的选择规则只用此前数据，往前走一年（参考 qlib `RollingGen` 的 `ROLL_EX` + `trunc_days`）。
9. 终点：封存测试段只在用户批准后一次性使用，不进循环。

**对比表**

| 项目 | 是否真带回测的研究循环 | 假设/记录结构 | 采纳判定 | 防过拟合 | 能否直接用于我们 |
|---|---|---|---|---|---|
| RD-Agent | 是 | Hypothesis（6 个字符串）+ Trace.hist/dag_parent | LLM 主观 yes/no，无阈值 | 弱；循环看 test 段 | 否（LLM API、Docker 倾向、口径不同），只学结构 |
| QuantaAlpha | 是 | StrategyTrajectory（含血缘） | 指标 + 质量门禁 | 复杂度/AST 冗余门禁；但有测试集反馈与 bug 报告 | 否，学数据结构与门禁 |
| LLM_QUANT_FACTORY/AutoAlpha | 是 | ResearchProposal + CandidateEvidence | 固定硬门禁 + DSR/PBO/bootstrap | 强：反馈限制、污染台账、冷却 | 否（重、需 LLM API），最值得学协议 |
| AlphaAgent | 部分（挖掘为可选） | DSL + FactorZoo | IC/换手报告 | 未读到 | 否，无许可证 |
| autoresearch | 通用循环 | results.tsv + git 提交 | 单指标变好即保留 | 无 | 形态最吻合，思想可直接用 |
| AI-Scientist | 通用科研 | 想法 JSON + 论文 | LLM 评审 | 无 | 否 |
| alphagen | RL 找因子，非假设循环 | 公式 | 组合 IC 奖励 | 无 | 否 |
| TradingAgents / ai-hedge-fund / FinRobot | 否（多角色决策聊天） | 对话/决策日志 | LLM 决策 | 前视修复为主 | 否 |
| qlib | 工具而非循环 | 任务配置 | — | RollingGen 时间切分 | 学切分逻辑，不直接用 |

**建议照搬的设计清单（9 条）**
1. 循环形态：编程助手 + 一份 program.md + git 分支 + 只追加的结果日志，评价代码只读不准改 — karpathy/autoresearch 的 `program.md`。
2. 角色边界：研究员（提假设）、评审员（查时间语义/重复/复杂度）、执行员（只跑已批准的）；评价阈值和代码在一个研究"世代"内冻结 — LLM_QUANT_FACTORY 的 `AutoAlpha/program.md`。
3. 假设必须落盘、可证伪、单一变更：机制、change、expected、falsification — AutoAlpha `program.md` 第 2 节，并吸收 RD-Agent `core/proposal.py` 的 `Hypothesis`、`reason` 字段。
4. 试验树 + 父子血缘：每条记录含 parent_ids、指标、反馈、失败类型 — QuantaAlpha `pipeline/evolution/trajectory.py` 的 `StrategyTrajectory`；RD-Agent `Trace.dag_parent`。
5. 反馈只回"哪道门没过 + 预设诊断"，不回离门槛的距离或封存段信息；每个方向限 3 次、连续 2 次无改善则停并冷却 — AutoAlpha `program.md`"Feedback policy"。
6. 采纳用固定规则而非 LLM 主观判断，并带多重检验意识（DSR、试验次数）— AutoAlpha 评价顺序（`program.md` 第 3 节）；反面教材是 RD-Agent `prompts.yaml` 的 `factor_feedback_generation`。
7. 因子准入前做复杂度与结构去重：符号长度、基础特征数、自由参数占比；与因子库做 AST 子树匹配 — QuantaAlpha `configs/experiment.yaml` 的 `quality_gate`、`factor.complexity`、`factor.duplication`。
8. 污染台账：任何一次碰到封存段的查看都要记录并使相关因子失去最终测试资格，不能靠换编号清零 — LLM_QUANT_FACTORY `INSTITUTIONAL_RESEARCH_CONTROLS.md`。
9. 去前视与时点：训练末端截断避免标签重叠（`trunc_days`）；信号形成后按下一开盘成交评价 — qlib `workflow/task/gen.py` 的 `RollingGen`；LLM_QUANT_FACTORY `FACTOR_LEAKAGE_AUDIT_20260716.md`。（我们引擎已按真实账户记账，这条主要用于核对。）

**不建议照搬**：RD-Agent 的"任何小改进都采纳"；用 test/封存段做循环反馈；LLM 自评打分；把 LLM 写代码的沙箱和 Docker 整套搬来。

### Gaps
- 本笔记没有设计具体门槛数值（需用户批准并先登记）；没有试跑任何仓库，所以"能否在 pandas 3 上运行"一律未验证。
