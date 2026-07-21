# 第4周：克制型 LSTM 训练结果

## 实验口径

- 时间边界、`target_h5/15/60` 和测试样本与 XGBoost 的冻结 split 文件完全一致，split 归属由 `target_time` 决定。
- 主输入为 10 个变量：历史 `token_load`、`request_count`、平均输入/输出 Token、GPT-4/API 占比、日/周正余弦；`--feature-set history_load` 可运行仅负载消融，本次主交付未额外运行消融。
- 三个窗口均冻结为 `lookback=144`（12 小时）、单层 `LSTM(32, dropout=0.1) → Dense(1)`，架构参数量 5,537。
- 每个窗口只正式运行 1 个预先声明的候选。训练采用 Adam、MAE、RMSE、`batch_size=64`、最多 100 epoch，以及 `EarlyStopping(patience=10, restore_best_weights=True)`。
- 特征 scaler 只拟合训练窗口使用的唯一输入行；目标 scaler 只拟合训练标签。验证/测试仅 `transform`。最终模型按验证集确定的最佳 epoch 在训练+验证序列上重训。
- 所有配置均在打开测试 split 前写入 YAML；每个最终模型只调用一次测试集 `predict`。输出逆标准化为 Token，并将负预测截断为 0。

## 验证集选择证据

| 预测窗口 | 最佳 epoch | 训练 MAE（Token） | 验证 MAE（Token） | 验证突发 F1 |
|---:|---:|---:|---:|---:|
| 5 分钟 | 42 | 29,998.654 | 19,831.100 | 0.5435 |
| 15 分钟 | 45 | 35,363.533 | 24,862.829 | 0.4034 |
| 60 分钟 | 19 | 47,106.206 | 35,291.219 | 0.1017 |

三个候选均优于验证集 seasonal-naive MAE（54,035.306）与对应窗口的 persistence；最佳 epoch 完全由验证 MAE 决定，因此按预案冻结小网络，不扩大搜索。

## 一次性测试结果

| 预测窗口 | LSTM MAE | Persistence MAE | Seasonal-naive MAE | XGBoost MAE | LSTM RMSE | 测试突发 F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 分钟 | 11,464.002 | 11,053.041 | 15,770.347 | 15,906.516 | 31,814.993 | 0.0000 |
| 15 分钟 | 11,054.480 | 12,342.419 | 15,770.347 | 16,464.424 | 30,904.246 | 0.0000 |
| 60 分钟 | 13,710.541 | 14,117.488 | 15,770.347 | 25,593.997 | 34,256.439 | 0.0000 |

LSTM 在三个窗口均优于 seasonal naive 与 XGBoost；在 15 和 60 分钟窗口优于 persistence，在 5 分钟窗口略逊于 persistence。训练集 P95 固定阈值为 315,246.2 Token/5min，而测试集仅有 5 个真实突发窗口；三个 LSTM 的测试 F1 均为 0。该结果反映固定绝对阈值下的跨时期分布偏移，不用于回调参数。

## 验收证据

- 三段形状与边界：`outputs/tables/table_05_lstm_sequence_audit.csv`
- 每段首条样本人工核对：`outputs/tables/table_05_lstm_manual_sequence_check.csv`
- 验证选择日志：`outputs/tables/lstm_tuning_log.csv`
- 逐 epoch 历史：`outputs/tables/lstm_history_h{5,15,60}.csv`
- 测试指标：`outputs/tables/table_05_lstm_results.csv`
- 测试预测：`outputs/tables/pred_lstm_test_h{5,15,60}.csv`
- 配置：`configs/lstm_h{5,15,60}.yaml`
- 模型、两个 scaler 与模型摘要：`models/lstm_*`
- 训练曲线：`outputs/figures/fig_lstm_training_h{5,15,60}.png`

三个 `.keras` 文件均已在独立 Python 进程中重新加载，架构参数量均核对为 5,537；9 条 split 审计和 9 条人工样本时间对齐检查全部通过。正式 h15 流程也已通过独立进程完整运行，未使用 Notebook 隐藏变量。

## 复现命令

```powershell
.\.venv\Scripts\python.exe src/06_lstm.py --horizon 15
.\.venv\Scripts\python.exe src/06_lstm.py --horizon 5 --horizon 60
```
