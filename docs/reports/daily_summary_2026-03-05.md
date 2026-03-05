# 今日工作汇总（2026-03-05）

## 1) 今日完成事项与关键结果
- 完成 M1 基线脚手架并稳定（toy pipeline 可运行）。
- 完成 GraphRAGBench 适配器首版：支持 train/val/test split 过滤。
- 修复 CLI 模式暴露问题：新增 `graphragbench-debug` 运行选项（commit `4023cc3`）。
- 增加实验追踪账本：run-id 维度 CSV/JSONL ledger（commit `63a674e`）。
- 完成 6 小时研究巡检并落盘：`docs/research/6h_patrol_2026-03-05.md`。

## 2) 进行中的实验与当前状态
- GraphRAGBench debug 冒烟：可跑通（avg_score=0.5, num_samples=2），用于流程连通性验证。
- local-debug 冒烟：可跑通（avg_score=1.0, num_samples=2）。
- M2 里程碑状态：
  - #1 适配器：已完成基础版，官方 metric parity 待补齐。
  - #6 实验追踪：run-id + CSV/JSONL 已落地，seed sweep 待实现。

## 3) 失败与风险
- 风险A：GraphRAGBench 评测仍为 fallback 方案（lexical-F1 + exact/contains），尚未与官方指标完全对齐。
- 风险B：测试工具链未完整（当前环境缺少 `pytest`），会影响自动回归执行效率。
- 风险C：retriever training loop / distributed launcher 尚未落地，可能影响后续可复现实验节奏。

## 4) 明日优先级 Top3
1. 完成 GraphRAGBench 官方 metric parity（评测口径对齐）。
2. 落地 retriever 训练循环（dataloader + checkpoint）。
3. 补齐 seed sweep 与回归测试，形成可稳定复现实验流水线。

## 5) 需要用户决策的点
- 优先级取舍：
  - 方案A：先 metric parity（论文/benchmark 对齐优先）；
  - 方案B：先 retriever training loop（模型训练能力优先）。
- 建议默认执行 A→B（先保证评测可信，再扩训练）。
