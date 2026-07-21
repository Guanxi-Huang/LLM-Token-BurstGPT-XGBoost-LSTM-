# 干净环境复跑与差异审计报告

复跑日期：2026-07-21；平台：Windows 11；Python 3.12.13；全局随机种子：42。

## 1. 干净状态与原始数据保护

- 在新的 PowerShell/Python 进程中，将原 `data/processed`、`models` 和 `outputs` 移至 `tmp/repro_backup_20260721_153000/`，随后从 `src/01_clean.py` 开始执行 README 的正式命令。
- 备份包含 15 个 processed 文件（108,129,537 B）、17 个模型文件（20,709,040 B）和 174 个输出文件（54,948,624 B）；`manifest_before.csv` 记录 207 个条目及可读取文件的 SHA-256。
- `outputs/tables/preliminary_model_metrics_h15.csv` 被另一个进程锁定，未能移动或读取。该 898 B 历史初步结果不被 `01`–`07`、论文主表或最终图册引用；最终流程继续执行并在此披露例外。
- `Dataset/` 中 10 个原始 CSV（合计 2,668,127,157 B）未删除、未移动、未改名。原始数据不进入 Git。

## 2. 复跑环境

| 组件 | 版本 |
|---|---:|
| Python | 3.12.13 |
| pandas | 3.0.1 |
| NumPy | 2.3.5 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.3.0 |
| TensorFlow | 2.20.0 |
| SHAP | 0.52.0 |

CPU 环境可完成全流程；GPU 不是复现前提。本机完整 `01`–`07` 的实测墙钟时间约 110 分钟，其中 LSTM 约 102 分钟。LSTM 运行期间发生两次异常长的系统/终端停顿，因此这个数字应视为本机保守上界，而非训练算法的稳定基准。其余主要步骤：清洗约 59 秒、聚合约 5 秒、特征约 3 秒、基线约 5 秒、XGBoost 约 5–6 分钟、统一评估约 40 秒；Week 5 消融与跨批次评估另行执行。

## 3. 时间契约修复及预期差异

干净复跑前的审计发现：时间戳表示 5 分钟窗口左边界 `[t,t+5min)`。因此在预测原点 `t`，刚完整结束且可使用的最后窗口必须标记为 `t-5min`，目标为 `t+h`。旧 Persistence 和 LSTM 分别读取了标记为原点的窗口，形成一个 5 分钟的未来信息偏移；XGBoost 已通过 `shift(1)` 遵守正确契约。

修复后：

- Persistence 使用 `target.shift(h/5+1)`；
- LSTM 的 144 步输入结束于预测原点前 5 分钟；
- 全部切分仍以目标时间为准；
- scaler、P95 和需要拟合的统计量仅来自训练范围；候选模型只按验证集选择。

因此，XGBoost、Seasonal Naive、P95 和切分结果与复跑前一致，而 Persistence、LSTM 及依赖它们的主表/稳健性表发生可解释的数值变化。15 分钟主测试 MAE 的主要差异如下。

| 模型 | 复跑前 MAE | 严格契约后 MAE | 解释 |
|---|---:|---:|---|
| Persistence | 11,901.6 | 12,342.4 | 输入向过去移动一个完整 5 分钟窗 |
| Seasonal Naive | 15,770.3 | 15,770.3 | 无变化 |
| XGBoost | 16,464.4 | 16,464.4 | 原实现已严格滞后 |
| LSTM | 10,727.6 | 11,054.5 | 序列末端向过去移动一个完整 5 分钟窗 |

以上差异是方法修复的预期结果，不应通过调整随机种子或测试集调参“恢复”旧值。最终论文只引用严格契约后的表图。

## 4. 最终关键结果

- 主数据：清洗后批次 1/2 共 5,058,910 条请求，聚合为 34,848 个 5 分钟窗；切分为 24,393/5,227/5,228。
- 固定容量阈值：训练集 `token_load` P95 = 315,246.2 Token/5min。
- 15 分钟主测试：LSTM MAE 11,054.5，低于 Persistence 12,342.4、Seasonal Naive 15,770.3 和 XGBoost 16,464.4。
- 主测试仅 5 个真实阈值突发窗，四模型固定阈值 F1 均为 0；XGBoost 的 AP 最高（0.0113），但不代表固定阈值检测已达到可部署水平。
- 批次 3 零样本 15 分钟测试中，Persistence 的 MAE 最低（43,838.2）且 F1 最高（0.711）；该排名变化说明跨时期分布漂移，不等同于在批次 3 内重训的结果。

## 5. 验收命令与状态

| 检查 | 状态 |
|---|---|
| `scripts/validate_week2_outputs.py` | PASS（10 张 EDA 图、13 个 Notebook 代码单元、聚合复算） |
| `scripts/validate_week3_outputs.py` | PASS（36 个无泄漏特征、36 行调参日志、3 个冻结 XGBoost） |
| `scripts/validate_week4_outputs.py` | PASS（LSTM 配置、序列、scaler、预测、训练图） |
| `scripts/validate_temporal_contract.py` | PASS（基线/LSTM 输入时刻、目标切分、训练拟合边界） |
| `scripts/validate_week5_outputs.py` | PASS（统一指标、消融边界、批次 3 零样本设计） |
| `scripts/validate_w5_visualizations.py` | PASS（Python/RStudio 双图册 20 张核心图） |

最终紧凑交付物由 `scripts/package_final_artifacts.py` 复制到 `artifacts/final/`，并在 `manifest_sha256.csv` 中记录源路径、字节数和 SHA-256。请求级数据、逐窗预测和模型二进制不进入该目录；必要时可按 README 从原始 CSV 重建。
