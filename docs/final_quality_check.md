# 最终质量检查

检查日期：2026-07-21。检查对象为干净复跑后冻结的 `artifacts/final/`、论文 DOCX/PDF、15 分钟演示稿、README、代码与实验日志。

| 检查项 | 状态 | 证据或处理 |
|---|---|---|
| 图轴有单位 | 通过 | 论文与演示逐页渲染检查；负载统一标注 `Token/5min`，误差、F1、AP 与比例轴均有名称。 |
| 表格说明数据集和预测窗口 | 通过 | 论文主表标题明确 BurstGPT 批次 1/2、15 分钟；稳健性表明确批次 3、零样本、15 分钟；附录表明确 5/60 分钟。 |
| 百分比与 Token 单位一致 | 通过 | MAE/RMSE 使用 `Token/5min`；sMAPE 和改善率使用百分比；阈值为 315,246.2 `Token/5min`。 |
| 无测试集调参 | 通过 | XGBoost 12 个候选只按验证 MAE 选择；LSTM 最佳 epoch 只按验证 MAE 选择；`validate_temporal_contract.py` 与 Week 3–5 校验通过。 |
| P95 只来自训练集 | 通过 | 固定阈值 `q0.95(y_train)=315,246.2`；同一值原样用于验证、测试和批次 3；时间契约校验通过。 |
| 未来信息不可用 | 通过 | 左边界时间契约为：预测原点 `t`、最后完整输入 `t-5min`、目标 `t+h`；所有切分按目标时间。 |
| README 文件名与命令 | 通过 | `01_clean.py` 至 `07_evaluate.py`、`03_features.py`、Week 5、论文/表图/演示生成器及产物路径均与仓库一致。 |
| 可追溯最终表图 | 通过 | `artifacts/final/manifest_sha256.csv` 记录 36 张表、60 张图的来源、字节数和 SHA-256。 |
| Python/RStudio 双图册 | 通过 | Week 5 共 22 张 Python/RStudio 图通过共同 12 行指标网格校验；共享 CSV 保留在最终表目录。 |
| 论文版式 | 通过 | Word 导出 14 页 PDF，逐页检查并修复分页、表格换行和图注；最终 DOCX/PDF 均保留。 |
| 演示版式与时长 | 通过 | 10 页逐页全尺寸检查，无裁切、重叠或未解析占位符；讲稿目标约 13 分 50 秒，预留约 1 分 10 秒。 |
| 原始数据安全 | 通过 | `Dataset/` 中 10 个原始 CSV 未删除、未移动、未提交；原始与衍生大文件均由 `.gitignore` 排除。 |
| 密钥与意外大文件 | 通过 | 提交前执行凭据模式扫描和未忽略文件大小审计；未发现私钥、API 密钥或意外大型文件。 |

## 自动校验结果

- `validate_week2_outputs.py`：通过；34,848 个窗口、5,058,910 条主批次请求、固定训练 P95 与人工窗口复算一致。
- `validate_week3_outputs.py`：通过；36 个特征、目标时间切分、调参日志、冻结模型、预测与解释图一致。
- `validate_week4_outputs.py`：通过；配置、scaler、训练历史、时序审计、预测和指标内部一致。
- `validate_temporal_contract.py`：通过；完成窗口输入、目标时间切分、训练集拟合和验证集选参一致。
- `validate_week5_outputs.py`：通过；主指标、分组误差、消融边界和批次 3 零样本设计一致。
- `validate_w5_visualizations.py`：通过；22 张双语工具链图和共同 12 行指标网格一致。
- `python -m compileall -q src scripts` 与 `node --check scripts/build_presentation.mjs`：通过。

## 已知非阻塞事项

备份生成物时，一个历史且未被论文、README 或最终清单引用的 `outputs/tables/preliminary_model_metrics_h15.csv` 被其他进程占用，未从原位置移动；正式复跑产物和最终结论不读取该文件。详情见 `docs/reproducibility_report.md`。
