# LLM 服务 Token 负载预测与突发流量识别

本仓库使用 BurstGPT v2.0 的真实请求轨迹，把请求级记录聚合为左闭右开的 5 分钟窗口，比较 Persistence、Seasonal Naive、XGBoost 与单层 LSTM 在未来 5、15、60 分钟 Token 负载预测及固定容量阈值突发识别上的表现。主问题和正文结果以 15 分钟为准；5/60 分钟为附录，BurstGPT 第 3 批为零样本跨时期稳健性检验。

## 1. 关键口径

- `total_tokens = Request tokens + Response tokens`；5 分钟负载为窗口内所有请求 `total_tokens` 之和，单位统一为 `Token/5min`。
- 每个时间戳是窗口左边界，聚合区间为 `[t, t+5min)`。在预测原点 `t`，最后可用输入窗口必须是 `t-5min`；目标时间为 `t+h`。
- 主数据为 BurstGPT `without_fails` 第 1、2 批；第 3 批不参与拟合或选择，只作外部零样本检验。
- 切分按目标时间连续划分 70%/15%/15%，不随机打乱。所有 scaler、训练 P95 阈值和需要拟合的统计量只用训练集；超参数只看验证集；测试集只用于冻结后的最终评价。
- 突发定义为 `token_load >= 训练集 token_load 的 P95`。`average precision (AP)` 使用非插值平均精度，不将其误称为梯形 PR-AUC。

## 2. 环境要求

已验证环境为 Windows 10/11、PowerShell 5.1+、Python 3.12、Git 2.45+。完整三批主 CSV 约 412 MB，清洗和建模建议至少 16 GB 内存、10 GB 可用磁盘；TensorFlow 在 CPU 上可以运行，GPU 不是必需条件。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import pandas, xgboost, tensorflow; print(pandas.__version__, xgboost.__version__, tensorflow.__version__)"
```

如果 PowerShell 阻止激活脚本，可不激活环境，直接把下文的 `python` 替换为 `.\.venv\Scripts\python.exe`。

## 3. 原始数据位置

从 [BurstGPT v2.0 release](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0) 获取数据，把以下文件原名放入 `Dataset/`：

```text
Dataset/
├─ level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv
├─ level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv
├─ level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv
└─ level_4_BurstGPT_v2.0_complete_1_Undisclosed_GitHub.csv
```

前三个文件分别用于主批次 1/2 和稳健性批次 3；`complete_1` 仅用于失败请求审计。原始 CSV、清洗后数据、模型和普通运行输出均由 `.gitignore` 排除，不应提交到版本库。

## 4. 从头复跑主流程

在仓库根目录按顺序执行。`03_features.py` 是模型共同依赖，不可跳过。

```powershell
.\.venv\Scripts\Activate.ps1
python src/01_clean.py
python src/02_build_series.py
python src/03_features.py
python src/04_baselines.py
python src/05_xgboost.py --horizons-minutes 5 15 60 --n-iter 12 --shap-sample-size 2000
python src/06_lstm.py --horizon 5 --horizon 15 --horizon 60 --max-runs 1
python src/07_evaluate.py --bootstrap-replicates 1000
```

主流程完成后运行独立验收：

```powershell
python scripts/build_eda_notebook.py
python scripts/validate_week2_outputs.py
python scripts/validate_week3_outputs.py
python scripts/validate_week4_outputs.py
python scripts/validate_temporal_contract.py
```

跨时期稳健性和 XGBoost 最小消融是可选扩展，依赖已冻结的主模型：

```powershell
python src/11_week5_experiments.py --force-clean --bootstrap-replicates 500
python src/12_visualize_w5_python.py
# 安装 R 4.5+ 与 ggplot2/scales/patchwork 后可选：
Rscript notebooks/04_w5_visualization_rstudio.R
python scripts/validate_temporal_contract.py
python scripts/validate_week5_outputs.py
python scripts/validate_w5_visualizations.py
```

## 5. 产物与追溯关系

| 步骤 | 主要输入 | 主要产物 |
|---|---|---|
| `01_clean.py` | `Dataset/*.csv` | `data/processed/requests_{1,2}.parquet`、数据质量表、清洗审计图 |
| `02_build_series.py` | 清洗后请求 | `series_5min.parquet`、切分表、训练 P95 阈值、人工窗口复算 |
| `03_features.py` | 5 分钟序列 | `features_h{5,15,60}_{train,valid,test}.parquet`、特征可用性表 |
| `04_baselines.py` | 5 分钟序列 | 验证/测试基线逐窗预测 |
| `05_xgboost.py` | 特征分片 | 验证选择日志、冻结配置、模型、测试预测、SHAP/重要性图 |
| `06_lstm.py` | 同一目标时间分片 | 验证选择日志、scaler、模型、测试预测、序列时间审计 |
| `07_evaluate.py` | 四模型冻结预测 | `table_04`–`table_06`、`fig_11`–`fig_18`、统一逐窗预测 |
| `11_week5_experiments.py` | 冻结主模型、第 3 批 | `table_06a`、`table_07`、第 3 批逐窗预测 |

运行中的完整结果在 `outputs/tables/`、`outputs/figures/`，模型在 `models/`，实验决策和冻结证据在 `docs/experiment_log.md` 与 `configs/*.yaml`。最终对外表图、论文和展示稿另存于 `artifacts/final/` 和 `deliverables/`，可以在不提交原始数据的情况下随 Git 版本追溯。

## 6. 预计运行时间与确定性

参考配置为 8 核桌面 CPU、16 GB 内存、无 GPU。首次安装通常需要 10–30 分钟；完整 `01`–`07` 在 CPU 上建议预留 60–120 分钟，其中 LSTM 占主要时间；`11_week5_experiments.py` 预计另需 5–20 分钟。本机干净复跑约 110 分钟，期间有两次异常系统停顿，逐步实测时间和差异原因记录在 `docs/reproducibility_report.md`。

全局随机种子为 42（Python、NumPy、XGBoost、TensorFlow）；TensorFlow 启用确定性运算并关闭 oneDNN 快速路径。若结果不同，依次核对：原始文件 SHA-256、Python/包版本、配置 YAML、随机种子、CPU/GPU 后端，以及是否从新的 PowerShell 和 Python 进程启动。不要用 Notebook 中的历史变量生成正式表图。

## 7. 论文、表图与展示稿

清洁复跑后，最终材料的生成命令为：

```powershell
python scripts/package_final_artifacts.py
python scripts/build_paper.py
```

PowerPoint 由 `scripts/build_presentation.mjs` 生成。该脚本依赖 Codex Desktop 随附的 `@oai/artifact-tool`；在已初始化该包的 ES-module 工作区中运行：

```powershell
Copy-Item scripts/build_presentation.mjs tmp/presentation_workspace/build_presentation.mjs -Force
node tmp/presentation_workspace/build_presentation.mjs `
  --project-root (Get-Location).Path `
  --out-dir ((Get-Location).Path + '\deliverables') `
  --preview-dir ((Get-Location).Path + '\tmp\presentation_preview')
```

若本机没有该运行时，可直接使用已交付的 `deliverables/LLM_Token_Load_Forecasting_15min.pptx` 和同名 `speaker_notes.md`；演示生成器不会参与模型结果计算。论文源稿位于 `paper/paper_draft.md`，Word/PDF 初稿位于 `deliverables/LLM_Token_Load_Forecasting_Draft.docx` 与 `deliverables/LLM_Token_Load_Forecasting_Draft.pdf`。

## 8. 数据许可与引用

BurstGPT 发布仓库标注为 CC BY 4.0；使用时应按其 release 和论文要求署名。仓库不再分发任何原始 CSV，也不提交可能包含请求级轨迹的衍生大文件。引用元数据集中维护在 `paper/references.bib`，论文参考文献中的 DOI/URL 经过逐项可访问性核验。

## 9. 常见问题

- 缺少 `series_5min.parquet`：先运行 `01_clean.py` 和 `02_build_series.py`。
- 缺少特征分片：运行 `03_features.py`，不要直接从基线跳到 XGBoost/LSTM。
- TensorFlow 安装失败：确认使用 64 位 Python 3.12，并缩短临时安装路径；详见 `docs/environment_status.md`。
- 正式结果与旧图不一致：先看 `docs/reproducibility_report.md`。本项目不以旧 Week 文稿覆盖新的干净复跑结果。
