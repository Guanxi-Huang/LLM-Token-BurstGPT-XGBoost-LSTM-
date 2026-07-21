# Week 4 可视化图册目录

W4 的 Python 与 RStudio 图册使用同一组经校验的 `outputs/tables/w4_visualization/` 共享 CSV。Python 脚本先从冻结配置、序列审计、训练历史、正式指标和对齐预测中导出绘图数据；RStudio 只读取这些共享 CSV，避免两个工具链出现数字口径差异。

## 图册位置

- Python：`outputs/figures/w4_python/`（9 张主题图 + 1 张总览图）
- RStudio：`outputs/figures/w4_rstudio/`（9 张主题图 + 1 张总览图）
- 共享绘图数据：`outputs/tables/w4_visualization/`

## 主题覆盖

| 编号 | 主题 | 主要证据 |
|---|---|---|
| W4-01 | LSTM 序列样本数与分段审计 | 三窗口 train/valid/test 序列规模、144 步回看、标签不越界 |
| W4-02 | 每步输入构成 | 1 项历史负载、5 项请求结构、4 项相对周期相位 |
| W4-03 | 训练与验证曲线 | 三窗口 scaled MAE、EarlyStopping 与最佳 epoch |
| W4-04 | 验证 MAE | Persistence、Seasonal Naive、XGBoost、LSTM 的验证比较 |
| W4-05 | 一次性测试 MAE | 冻结后四类方法的三窗口测试比较 |
| W4-06 | 15 分钟代表性轨迹 | 同一完整测试日的实际值与四类预测 |
| W4-07 | 15 分钟测试均值偏差 | 实际均值、预测均值及偏差方向 |
| W4-08 | 固定训练 P95 的突发结果 | 真实/预测突发窗数量与 F1=0 限制 |
| W4-09 | 冻结训练概况 | 训练/验证 MAE、最佳 epoch、统一 5,537 参数架构 |

## 运行顺序

```powershell
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\validate_week4_outputs.py
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\validate_temporal_contract.py
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' src\12_visualize_w4_python.py
& 'E:\R\R-4.5.2\bin\Rscript.exe' --vanilla notebooks\04_w4_visualization_rstudio.R
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\validate_w4_visualizations.py
& 'C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\write_week4_results_to_docx.py
```

只有前两项验收均通过后，图册和 Word 结果才可作为正式 W4 证据。旧版缺少 `input_end_time` 或未满足 `feature_time - input_end_time = 5min` 的结果不得复用。

## Word 取图

独立 W4 Word 文稿从完整双工具链图册中选取 8 张互补图：Python 4 张、RStudio 4 张。其余主题图和两张总览图保留在上述目录，便于复核而不挤压正文版面。
