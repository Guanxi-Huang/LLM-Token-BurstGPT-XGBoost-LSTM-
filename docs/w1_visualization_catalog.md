# W1结果可视化图册：Python＋RStudio

本图册把W1的字段核验、分块清洗、质量审计、批次边界、两天冒烟测试、文献矩阵和清洗口径下的初步模型结果整理为两套可复现图形。Python负责读取Parquet并生成统一的绘图中间表；RStudio只读取这些中间CSV，从而保证两种工具使用相同分母、清洗规则和时间范围。

## 1. 总览

### Python总览

![Python W1 visualization overview](../outputs/figures/w1_python/w1_python_overview.png)

### RStudio总览

![RStudio W1 visualization overview](../outputs/figures/w1_rstudio/w1_rstudio_overview.png)

## 2. 图形索引与可支持结论

| 编号 | 分析问题 | Python图 | RStudio图 | 图形可支持的结论 |
|---|---|---|---|---|
| W1-01 | 清洗保留了多少记录？ | `fig_w1_py_01_cleaning_retention.png` | `fig_w1_r_01_cleaning_retention.png` | 三个批次分别删除2.91%、2.34%、3.04%；所有删除均为全字段重复。`complete_1`零输出失败率为1.78%。 |
| W1-02 | 真实字段能否支持计划特征？ | `fig_w1_py_02_field_support.png` | `fig_w1_r_02_field_support.png` | 1/2号没有Session、Elapsed time、独立调用方式或真实日历；3号Session/Elapsed time只可审计，不能扩展主模型。 |
| W1-03 | 三个批次的时间关系如何？ | `fig_w1_py_03_batch_time_coverage.png` | `fig_w1_r_03_batch_time_coverage.png` | 1/2号近似连续，适合主实验；3号在约104天间隔后开始，只用于跨时期稳健性测试。 |
| W1-04 | 请求量是否平稳？ | `fig_w1_py_04_daily_requests.png` | `fig_w1_r_04_daily_requests.png` | 主实验日请求量存在明显水平变化、尖峰和低活跃期，不能把整个121天视为同分布样本。 |
| W1-05 | 单请求Token是否长尾？ | `fig_w1_py_05_token_distribution.png` | `fig_w1_r_05_token_distribution.png` | 1/2号均为长尾分布且形态不同；极端值和批次漂移需要在稳健性分析中单独报告。 |
| W1-06 | 模型与日志类型构成是否稳定？ | `fig_w1_py_06_category_composition.png` | `fig_w1_r_06_category_composition.png` | 2号的ChatGPT/API占比明显高于1/3号，跨批次误差可能受服务构成漂移影响。 |
| W1-07 | 两天端到端聚合是否合理？ | `fig_w1_py_07_smoke_profile.png` | `fig_w1_r_07_smoke_profile.png` | 576个完整5分钟窗口覆盖6,107条请求、5,694,880 Token；请求数与Token曲线同步但并非固定比例。 |
| W1-08 | 指定文献覆盖哪些证据主题？ | `fig_w1_py_08_literature_themes.png` | `fig_w1_r_08_literature_themes.png` | 文献组同时覆盖负载画像、forecasting、burst/SLO、服务系统、算法基础和基线评价；标记表示主题覆盖，不代表质量评分。 |
| W1-09 | 初步模型与基线表现如何？ | `fig_w1_py_09_preliminary_models.png` | `fig_w1_r_09_preliminary_models.png` | XGBoost测试MAE最低，但验证期不优于持久性；固定训练P95在测试期导致三种方法F1均为0。LSTM在W1尚不可用。 |

Python图目录：`outputs/figures/w1_python/`

RStudio图目录：`outputs/figures/w1_rstudio/`
两套图共享的绘图数据：`outputs/tables/w1_visualization/`

## 3. 复现方式

### Python

在项目根目录执行：

```powershell
& .\.venv\Scripts\python.exe src\09_visualize_w1_python.py
```

脚本读取1/2号清洗Parquet、W1质量表、两天冒烟CSV和文献矩阵编码，生成9张PNG、Python总览图及供R使用的CSV。

### RStudio

在RStudio中打开`notebooks/02_w1_visualization_rstudio.R`并点击 **Source**；也可从命令行执行：

```powershell
& E:\R\R-4.5.2\bin\Rscript.exe notebooks\02_w1_visualization_rstudio.R
```

脚本仅依赖当前已经安装的`ggplot2`、`readr`、`scales`和R自带`grid`。它会生成9张RStudio PNG、RStudio总览图和`r_session_info.txt`。

## 4. 解释边界

1. 1/2号日请求量与Token分布来自清洗后Parquet；3号未生成训练Parquet，因此只在质量、字段、时间范围和构成图中使用审计汇总，未虚构3号请求级曲线。
2. 文献主题矩阵由`docs/literature_matrix.md`人工编码，只用于展示证据覆盖，不能据其比较论文质量或效果大小。
3. W1-09是清洗口径下的初步XGBoost—基线结果；TensorFlow/Keras仍不可用，因此图中明确不包含LSTM。
4. 固定P95的零F1反映测试期事件极度稀疏和阈值失配，不能单独解释为模型完全没有预测能力。
