# 实验日志

记录每一次正式运行。探索性 Notebook 可以快速试验，但进入论文的数字、表格和图必须在此留下可追溯记录。

| 日期 | 阶段/脚本 | 配置或提交号 | 输入数据 | 结果与产物 | 决定与下一步 |
|---|---|---|---|---|---|
| 2026-07-20 | 第0天环境准备 | `configs/base.yaml`；seed=42 | 本地 BurstGPT 1/2/3 原始 CSV | 创建目录、脚本骨架、Python 3.12 虚拟环境；验证数据/统计/XGBoost/SHAP/Notebook 内核依赖 | 第1周开始核验实际字段并编写清洗脚本；TensorFlow 2.20.0待网络可用时补装，详见`environment_status.md` |
| 2026-07-20 | 初步描述统计与XGBoost | `src/08_preliminary_results.py`；seed=42；15分钟预测；固定70/15/15时间切分 | `BurstGPT_v2.0_without_fails_1/2` | 5,188,507条请求聚合为34,848个5分钟窗口；XGBoost测试MAE=11,958.6 Token/5min，较持久性/季节朴素基线低19.2%/35.5%；训练P95阈值下测试F1均为0 | 作为“描述统计＋XGBoost—基线”初步结果写入文稿；固定阈值的跨时段失配需讨论；LSTM待TensorFlow可用后按相同切分补跑 |

| 2026-07-21 | 第1周字段核验、分块清洗与质量审计 | `src/01_clean.py`；`chunksize=200_000`；命令 `& .\.venv\Scripts\python.exe src/01_clean.py` | 三个 `without_fails`；`complete_1` 仅作失败审计 | 1号 1,404,294→1,363,384（删40,910，2.9132%）；2号 3,784,213→3,695,526（删88,687，2.3436%）；3号 4,956,058→4,805,535（删150,523，3.0372%）；所有删除均为全字段重复，时间/负Token/关键缺失删除均为0。`complete_1` 零输出25,443/1,429,737=1.7796%。生成1/2号Parquet、质量/删除表、两张审计图；连续两天冒烟样本576个5分钟窗口、6,107条请求、5,694,880 Token。Notebook由 `nbclient` 全部7个代码单元无错误执行。 | 1/2用于主实验，3仅跨时期稳健性；`complete_1`不混入训练。真实时间为相对秒，删除真实星期/周末/节假日特征；1/2无Session，删除Session；无独立调用方式，以`Log Type`映射；`Elapsed time`仅3号且为事后量，不用。无请求ID使“重复删除”可能低估并发负载，保留为局限并计划做保留/删除敏感性分析。研究问题固定为3项，模型不再扩展。 |

| 2026-07-21 | 清洗口径下重跑描述统计与XGBoost—基线 | `src/08_preliminary_results.py`；seed=42；15分钟预测；固定70/15/15时间切分；训练集P95=315,900.2 | `data/processed/requests_1.parquet`、`requests_2.parquet`，共5,058,910条清洗后请求 | 34,848个5分钟窗口、累计2,146,868,147 Token。XGBoost测试MAE=10,029.8、RMSE=31,134.2，较持久性/季节朴素基线改善16.4%/36.9%；验证MAE比持久性差7.2%。测试期P95=65,411.3，仅为训练阈值20.7%；固定阈值下仅4个真实事件，三种方法F1均为0。结果保存为`preliminary_cleaned_*`表和图，并写入`results_discussion_preliminary.md`及清洗版Word文稿。 | 旧初步结果基于未去重CSV，仅保留为历史运行，不再作为正文数字。固定阈值适合绝对容量上限；相对突发需另行验证只依赖历史的滚动阈值且不得用测试标签调参。TensorFlow/Keras仍不可用，LSTM结果保持空缺，待环境可用后按完全相同切分、阈值和指标补跑。 |

| 2026-07-21 | W1全结果Python＋RStudio可视化 | `src/09_visualize_w1_python.py`；`notebooks/02_w1_visualization_rstudio.R`；Python 3.12；R 4.5.2 | 清洗后1/2号Parquet、W1质量/边界/冒烟/文献矩阵及清洗口径初步模型表 | Python生成9张图＋1张总览，RStudio生成9张图＋1张总览；共享绘图CSV保存于`outputs/tables/w1_visualization/`，图册见`docs/w1_visualization_catalog.md`。两套图覆盖清洗、字段、时间边界、日请求量、Token长尾、构成、冒烟、文献主题与初步模型结果。 | 3号没有主训练Parquet，因此只展示其审计汇总，不虚构请求级曲线；文献主题为人工证据编码而非评分；LSTM仍不进入W1模型图。R启动配置请求Windows不存在的`C.UTF-8`，脚本显式切换为中文UTF-8区域设置后成功运行。 |

| 2026-07-21 | W1结果写入Week2文稿 | `scripts/insert_w1_results_before_week2.py` | `LLM_Token_Load_Forecasting_Research_Plan_week2_results.docx`、W1验证结果与9张Python图 | 生成`LLM_Token_Load_Forecasting_Research_Plan_week2_results_with_W1.docx`；在原Week2之前新增第7章7个小节和图7-1至图7-9，原Week2顺延为第8章且表图题同步改为8章编号。 | 保留源文稿不覆盖；XGBoost—基线只标为初步结果，明确验证集不稳定与固定训练P95跨时段失配；LSTM继续留空，待TensorFlow可用后按同一切分、阈值和指标补跑。 |

## 记录要求

- 每次正式训练记录随机种子、配置文件、输入数据文件、运行命令和输出文件名。
- 每次使用验证集选择参数时，记录候选参数、验证指标和选择理由。
- 测试集仅用于最终汇报；不得据此继续选择特征、参数或模型。
- 若重新运行得到不同结果，记录差异、排查过程和最终处理。

<!-- WEEK2_AGGREGATION_CHECK_START -->

## 第2周：5分钟目标序列、人工核验与固定突发阈值

- 正式运行日期：2026-07-21；命令：`& .\.venv\Scripts\python.exe scripts/run_week2_pipeline.py`；配置：`configs/base.yaml`（seed=42）。
- 窗口语义：左闭右开 `[start, start+5min)`；公开时间仅为相对秒，表中的1970日期只是合成UTC轴。
- 切分：train=24,393行；valid=5,227行；test=5,228行。训练集P95固定阈值为 `315246.200000` Token/5min，三段统一按 `token_load >= threshold` 标记突发。
- 批次边界：1号最后请求到2号第一请求间隔 `441` 秒；两者所在窗口之间有 `1` 个完整窗口，其中 `1` 个为零负载。
- 以下三个窗口由seed=42从完整网格无放回抽样，再回到两个请求级Parquet用布尔条件逐项重算；均值在零请求窗保持NaN。

| 窗口起点（合成UTC） | 窗口终点（不含） | 原始Token | 聚合Token | 原始请求数 | 聚合请求数 | 原始平均输入 | 聚合平均输入 | 原始平均输出 | 聚合平均输出 | 原始GPT-4数 | 聚合GPT-4数 | 原始API数 | 聚合API数 | 一致 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1970-01-11T19:10:00+00:00 | 1970-01-11T19:15:00+00:00 | 4838 | 4838 | 4 | 4 | 895.500000 | 895.500000 | 314.000000 | 314.000000 | 2 | 2 | 0 | 0 | True |
| 1970-03-21T04:50:00+00:00 | 1970-03-21T04:55:00+00:00 | 140580 | 140580 | 557 | 557 | 244.152603 | 244.152603 | 8.235189 | 8.235189 | 0 | 0 | 556 | 556 | True |
| 1970-04-04T15:30:00+00:00 | 1970-04-04T15:35:00+00:00 | 14644 | 14644 | 8 | 8 | 1675.000000 | 1675.000000 | 155.500000 | 155.500000 | 1 | 1 | 0 | 0 | True |

结论：三个窗口的Token总量、请求数、输入/输出Token均值、GPT-4请求数和API请求数全部一致；窗口边界与零窗构造通过人工复算。

<!-- WEEK2_AGGREGATION_CHECK_END -->

<!-- LSTM_H15_START -->
### LSTM h15 (main)

- Config: `configs/lstm_h15.yaml`
- Architecture: Input → LSTM(32, dropout=0.1) → Dense(1)
- Lookback: 144 × 5 minutes
- Parameters: 5,537
- Best epoch: 45
- Train MAE: 35,363.533 Token
- Validation MAE: 24,862.829 Token
- Validation burst F1: 0.4034
- Seed (Python/NumPy/TensorFlow): 42
<!-- LSTM_H15_END -->

<!-- LSTM_H5_START -->
### LSTM h5 (main)

- Config: `configs/lstm_h5.yaml`
- Architecture: Input → LSTM(32, dropout=0.1) → Dense(1)
- Lookback: 144 × 5 minutes
- Parameters: 5,537
- Best epoch: 42
- Train MAE: 29,998.654 Token
- Validation MAE: 19,831.100 Token
- Validation burst F1: 0.5435
- Seed (Python/NumPy/TensorFlow): 42
<!-- LSTM_H5_END -->

<!-- LSTM_H60_START -->
### LSTM h60 (main)

- Config: `configs/lstm_h60.yaml`
- Architecture: Input → LSTM(32, dropout=0.1) → Dense(1)
- Lookback: 144 × 5 minutes
- Parameters: 5,537
- Best epoch: 19
- Train MAE: 47,106.206 Token
- Validation MAE: 35,291.219 Token
- Validation burst F1: 0.1017
- Seed (Python/NumPy/TensorFlow): 42
<!-- LSTM_H60_END -->

<!-- WEEK5_FINAL_EVALUATION_START -->
## Week 5: frozen unified test evaluation

- Freeze rule: baseline, XGBoost, and LSTM configurations were selected using validation evidence only; the test set is reporting-only.
- Common coverage: 62,736 long-form rows; four models at every timestamp+horizon; at least 5,228 common target timestamps per horizon.
- Burst rule: one train-only P95 = 3.15e+05 Token/5-min; labels use token_load >= threshold for every model and dataset.
- Confidence information: 95% synthetic-UTC calendar-day block bootstrap, seed=42, 1,000 replicates.
- Test-use rule: src/07_evaluate.py reads frozen predictions and does not refit, select, or tune a model.

### Main 15-minute results (three significant digits)

| Model | MAE | RMSE | sMAPE | vs Seasonal MAE | F1 | AP | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1.23e+04 | 3.72e+04 | 74.8% | 21.7% | 0 | 0.00537 | 0/5/5/5218 |
| Seasonal Naive | 1.58e+04 | 4.19e+04 | 91.4% | 0% | 0 | 0.00684 | 0/4/5/5219 |
| XGBoost | 1.65e+04 | 3.47e+04 | 130% | -4.4% | 0 | 0.0113 | 0/2/5/5221 |
| LSTM | 1.11e+04 | 3.09e+04 | 88.1% | 29.9% | 0 | 0.0108 | 0/0/5/5223 |

- Overall result: LSTM has the lowest MAE (1.11e+04 Token/5-min; 29.9% versus Seasonal Naive).
- Burst result: 5 actual burst windows occur in the test set. Best fixed-threshold F1 is Persistence, Seasonal Naive, XGBoost, LSTM (tie) (0); best ranking AP is XGBoost (0.0113).
- Segment result: non-burst MAE is lowest for LSTM; burst-window MAE is lowest for XGBoost. Target-time GPT-4/API shares are used only for retrospective grouping.
- Composition cutoffs: train nonzero-window medians are GPT-4=0.154, API=0.286.
- Relative day/week phases describe the anonymized synthetic time axis and are not interpreted as real weekdays or weekends.

### Figure conclusions

- Fig. 11: the objectively selected representative week (1970-04-19 to 1970-04-25) compares ordinary-load tracking under one common unit and legend.
- Fig. 12: the local window is centered on the maximum observed test target (1970-05-01T16:00:00+00:00) and shows every forecast against the train P95.
- Fig. 13: LSTM leads overall MAE, while the fixed-threshold F1 result must be interpreted separately.
- Fig. 14: XGBoost has the highest AP, which is ranking evidence rather than proof of adequate fixed-threshold detection.
- Fig. 15: confusion matrices expose TP/FP/FN/TN counts and the effect of rare positive windows.
- Fig. 16: burst-window errors are reported separately from non-burst errors rather than hidden by the overall average.
- Fig. 17: relative phase profiles are diagnostic positions on the synthetic axis, not real calendar semantics.
- Fig. 18: service-composition groups explain target-time errors only; they are not deployable future composition features.

<!-- WEEK5_FINAL_EVALUATION_END -->

<!-- WEEK5_ABLATION_ROBUSTNESS_START -->
## 第5周：最小消融与BurstGPT_3跨时期稳健性

### 15分钟XGBoost最小消融

- 三组仅为预先定义的完整特征、去除6个GPT-4/API历史占比特征、仅8个滞后+4个相对日历相位；全部使用 `configs/xgb_h15.yaml` 的固定参数和不变目标时间边界。
- 方案排序只看验证MAE；测试集只汇报三组最终结果。完整特征测试结果复用主预测，两个缩减组各发生1次最终测试预测，未据测试结果改动任何配置。

| Feature set | Features | Validation MAE | Test MAE | Test improvement vs Seasonal Naive | Test F1 |
|---|---:|---:|---:|---:|---:|
| full_features | 36 | 2.39e+04 | 1.65e+04 | -4.4% | 0 |
| without_service_structure | 30 | 2.5e+04 | 1.53e+04 | 3.21% | 0 |
| lags_plus_relative_calendar | 12 | 2.65e+04 | 1.39e+04 | 12% | 0 |

消融结论：验证集排序首位为 `full_features`；该排序不回写或替换已冻结的论文主模型。测试结果用于说明特征组贡献，不作为再次选择依据。

### BurstGPT_3稳健性设计（唯一选定方案）

- 设计：**外部零样本跨时期测试**。1/2号上冻结的Persistence、Seasonal Naive、XGBoost、LSTM及scaler直接应用于3号数据；3号内部不训练、不验证、不选模型。
- 清洗：复用 `src/01_clean.py::clean_and_audit_without_fails`；原始文件 `level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv`，清洗后 4,805,535 条请求，保存 `data/processed/requests_3.parquet`。
- 聚合：复用 `src/02_build_series.py::aggregate_complete_grid` 的左闭右开5分钟规则；单独得到 31,680 个窗口（1970-08-14T00:00:00+00:00 至 1970-12-01T23:55:00+00:00），不与1/2号拼接。
- 阈值与特征：继续使用1/2号训练期P95=315246.2 Token/5min；相对日/周相位锚定公开相对秒数的合成UTC原点。3号新增的Session ID与Elapsed time完全不用。
- 置信信息：95%合成UTC日块Bootstrap，seed=42，500次重复。

#### BurstGPT_3 15分钟结果（3位有效数字）

| Model | MAE | RMSE | sMAPE | Seasonal-Naive improvement | Precision | Recall | F1 | PR-AUC/AP | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 4.38e+04 | 1.42e+05 | 65.4% | 63.5% | 0.711 | 0.711 | 0.711 | 0.671 | 1409/573/573/28834 |
| Seasonal Naive | 1.2e+05 | 2.82e+05 | 108% | 0% | 0.0376 | 0.0378 | 0.0377 | 0.0683 | 75/1918/1907/27489 |
| XGBoost | 6.18e+04 | 1.63e+05 | 87.1% | 48.6% | 0.586 | 0.449 | 0.509 | 0.475 | 890/628/1092/28779 |
| LSTM | 5.35e+04 | 1.57e+05 | 77.7% | 55.5% | 0.807 | 0.383 | 0.52 | 0.625 | 760/182/1222/29225 |

稳健性结论：3号零样本15分钟总体MAE最低的是Persistence（4.38e+04 Token/5min）；该结果只检验跨时期迁移，不等同于在3号内部重训后的可达到性能。突发结果仍以固定1/2号训练P95为容量阈值解释。

### 测试使用次数补充

- 主测试集新增预测调用仅有两次：两个预定义XGBoost缩减特征组各一次；完整组复用冻结预测。消融测试标签从未参与排序。
- BurstGPT_3属于独立跨时期数据，不计入1/2号主测试集使用次数；每个冻结模型/窗口仅作一次零样本预测。
- 第5周在此停止调参与阈值修改；后续只允许修复代码错误、核对表图或撰写论文。

<!-- WEEK5_ABLATION_ROBUSTNESS_END -->
