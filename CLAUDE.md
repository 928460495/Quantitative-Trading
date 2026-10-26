# 项目说明（给 Claude Code）

A 股短期选股因子研究。研究流程和每个决定见 `README.md`、`docs/DECISIONS.md`。

## 沟通

- 用清楚直白的中文，不堆术语，不为显得专业用生僻词。
- 需要用户拍板的事：先自己把事实查清楚，给出推荐和理由，等用户确认再动手。

## 工作流程：superpowers

仓库自带 superpowers 技能（`.claude/skills/`，来源和许可见 `.claude/skills/README.md`）。**开始任何工作前先读 `.claude/skills/using-superpowers/SKILL.md`**（官方插件用启动钩子注入它，这里用这句话代替）。

本项目的用法，优先于技能里的默认做法：

- 新功能、新的策略规则、改研究设计：先用 brainstorming。策略本身的设计写进 `docs/STRATEGY.md`（策略说明书）；其他设计文档放 `docs/superpowers/specs/`。
- 设计批准后用 writing-plans 写计划，用 executing-plans 在当前会话里执行。除非用户明确要求，不用 subagent-driven-development、dispatching-parallel-agents，brainstorming 的 builder check 也由自己来读，不开子代理。
- test-driven-development 用在 `qt/` 里的库代码；`scripts/` 里的研究脚本，关键逻辑有测试即可。
- 查问题用 systematic-debugging；说"做完了"之前用 verification-before-completion。
- 不用 brainstorming 的可视化页面（visual companion）：云端会话里用户打不开本地浏览器页面。遥测已在 `.claude/settings.json` 关掉。
- 不用 using-git-worktrees：每个功能一个分支、一个 PR 的现行做法不变。

## 研究规矩（比任何技能都优先）

- 研究上的决定以 `docs/DECISIONS.md` 为准，改动要记日期和原因。
- 看结果之前，先把规则写进 `docs/DECISIONS.md` 并提交。
- 每次检验、合成、参数尝试都记入试验日志 `reports/trials.jsonl`。
- 最终测试段（2022 年至今）封存，只有最后一次最终测试才设 `QT_UNLOCK_TEST=1`。
- 到审核节点停下来等用户确认。
- 提交信息、PR、代码注释里不写模型名称。

## 常用命令

```bash
pytest -q                 # 全部测试
pytest -q -m "not data"   # 不需要数据包的部分
```
