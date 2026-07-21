# 真实LLM服务的Token负载预测与突发流量识别

## 基于BurstGPT真实轨迹的XGBoost与LSTM比较

这项研究以真实LLM服务轨迹为基础，围绕短期Token负载预测和突发流量识别开展实证分析。研究重点不是堆叠复杂模型，而是保证时间切分、基线比较、突发识别和可复现性严谨。

---

## 1. 研究方法与对应工具

### 1.1 研究对象与任务定义

将请求级轨迹聚合成5分钟时间序列：

`Y_t = Σ(Request Tokens_i + Response Tokens_i), i ∈ t`

第1周结束后固定以下三个研究问题：

- **RQ1（负载特征）：** BurstGPT 1/2号成功请求聚合成5分钟序列后，Token总负载呈现怎样的时间变化、分布与突发特征？
- **RQ2（预测比较）：** 在未来5、15、60分钟预测中，XGBoost与LSTM相对Persistence和Seasonal Naive基线的误差表现如何（15分钟为主任务）？
- **RQ3（突发与稳健性）：** 将预测负载与仅由训练集确定的P95阈值比较时，各方法识别突发的能力如何，结论能否在只作跨时期测试的3号批次保持？

研究问题与模型集合至此冻结：学习模型仅为XGBoost和LSTM，基线仅为Persistence与Seasonal Naive；不再增加模型种类。GPT-4与API占比只用于已有字段支持的分组误差分析，不新增研究问题。

### 1.2 方法-工具对应表

| 环节 | 具体方法 | 工具 |
|---|---|---|
| 数据读取 | CSV分块读取、类型压缩 | Python、Pandas；数据较大时使用Polars |
| 数据质量检查 | 缺失值、重复记录、零响应Token、时间间隔检查 | Pandas、NumPy |
| 时间序列构建 | 请求级数据聚合为5分钟时间窗 | `resample()`、`groupby_dynamic()` |
| 描述统计 | 趋势、周期、分布、峰谷比、模型构成 | Matplotlib、Seaborn |
| 周期分析 | 相对24小时/7天相位、自相关、周期图 | statsmodels、SciPy |
| 特征工程 | 滞后、滚动统计、周期编码、模型类型占比 | Pandas、scikit-learn |
| 简单基线 | Persistence、24小时季节朴素预测 | NumPy、scikit-learn |
| 机器学习 | XGBoost回归 | `xgboost.XGBRegressor` |
| 深度学习 | 单层或双层LSTM | TensorFlow/Keras |
| 参数优化 | RandomizedSearch、早停；Optuna可选 | scikit-learn、Optuna |
| 突发识别 | 训练集P95阈值、预测负载转为突发标签 | NumPy、scikit-learn |
| 预测评价 | MAE、RMSE、sMAPE | scikit-learn |
| 突发评价 | Precision、Recall、F1、PR-AUC | scikit-learn |
| 可解释性 | 特征重要性、SHAP值 | SHAP |
| 显著性与稳健性 | 配对误差检验、分块Bootstrap | SciPy、statsmodels |
| 实验管理 | 固定随机种子、保存参数和结果 | Git、GitHub、YAML、Joblib |

### 1.3 推荐特征

- 滞后负载：`lag_1、2、3、6、12、24、48、288`。
- 滚动特征：过去15、30、60分钟的均值、标准差、最大值。
- 周期特征：由相对秒数构造24小时与7天周期的正余弦相位。因采集日历未公开，不构造真实星期、周末、节假日等日历特征。
- 负载结构：请求数量、平均输入Token、平均输出Token。
- 服务结构：GPT-4请求占比、API请求占比；其中API/Conversation来自唯一的`Log Type`字段，不虚构独立“调用方式”字段。
- 变化特征：Token负载一阶差分、短期增长率。

注意：1/2号主实验文件没有`Session ID`和`Elapsed time`，因此删除活跃Session数、每Session平均请求数和响应耗时特征。未来请求的Response Tokens和Total Tokens在预测时不可知，只能作为预测目标或历史滞后特征。所有标准化、突发阈值和特征选择都必须只用训练集拟合。

### 1.4 实验设计

推荐按时间顺序切分，不能随机打乱：

- 训练集：前70%。
- 验证集：中间15%。
- 测试集：最后15%。
- BurstGPT_3：作为额外的跨时期稳健性测试。

对XGBoost和LSTM分别训练5、15、60分钟预测模型，15分钟作为主要结果。由预测负载与训练集P95阈值比较得到突发预测标签，从而避免再引入一套复杂分类模型。

不建议在六周内加入Transformer、Prophet、Random Forest和大量集成模型。它们会稀释论文主线；一项严谨的XGBoost-LSTM比较比模型数量更多但评价不完整更有说服力。

---

## 2. 相关文献15篇

### LLM工作负载与服务系统

[1] Wang, Y., et al. (2025). [BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems](https://doi.org/10.1145/3711896.3737413). *KDD 2025*。

用途：核心数据来源，说明真实LLM流量的周期性、异质性和突发性。

[2] Cortez, E., et al. (2017). [Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms](https://www.microsoft.com/en-us/research/publication/resource-central-understanding-predicting-workloads-improved-resource-management-large-cloud-platforms/). *SOSP 2017*。

用途：云计算工作负载预测与资源管理的理论背景。

[3] Shahrad, M., et al. (2020). [Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider](https://www.usenix.org/conference/atc20/presentation/shahrad). *USENIX ATC 2020*。

用途：生产环境请求流量的长尾、突发和周期特征。

[4] Yu, G.-I., et al. (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). *OSDI 2022*。

用途：说明LLM动态批处理、调度和请求负载之间的联系。

[5] Kwon, W., et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). *SOSP 2023*。

用途：说明Token数量、KV Cache和系统吞吐量之间的关系。

[6] Li, Z., et al. (2023). [AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan). *OSDI 2023*。

用途：直接讨论突发工作负载、模型并行和统计复用。

[7] Patel, P., et al. (2024). [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677). *ISCA 2024*。

用途：解释输入Token与输出Token对应的Prefill、Decode负载差异。

[8] Zhong, Y., et al. (2024). [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin). *OSDI 2024*。

用途：进一步说明Prefill与Decode资源需求不同。

[9] Stojkovic, J., et al. (2025). [DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency](https://www.microsoft.com/en-us/research/publication/dynamollm-designing-llm-inference-clusters-for-performance-and-energy-efficiency/). *HPCA 2025*。

用途：连接负载变化、动态资源配置、性能和能耗。

### 预测模型与评价方法

[10] Chen, T., & Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *KDD 2016*。

用途：XGBoost模型的原始方法文献。

[11] Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation, 9*(8), 1735-1780。

用途：LSTM的原始方法文献。

[12] Salinas, D., et al. (2020). [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://doi.org/10.1016/j.ijforecast.2019.07.001). *International Journal of Forecasting, 36*(3), 1181-1191。

用途：深度循环网络在时间序列预测中的代表性应用。

[13] Lim, B., et al. (2021). [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012). *International Journal of Forecasting, 37*(4), 1748-1764。

用途：多步预测和可解释时间序列模型的拓展文献。

[14] Oreshkin, B. N., et al. (2020). [N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting](https://arxiv.org/abs/1905.10437). *ICLR 2020*。

用途：深度时间序列预测的代表性基准。

[15] Makridakis, S., et al. (2018). [The M4 Competition: Results, Findings, Conclusion and Way Forward](https://doi.org/10.1016/j.ijforecast.2018.06.001). *International Journal of Forecasting, 34*(4), 802-808。

用途：支持设置季节朴素基线，避免默认复杂模型一定优于简单方法。

---

## 3. 可使用的数据集

| 数据集 | 时间及内容 | 论文用途 | 推荐程度 |
|---|---|---|---|
| [BurstGPT v2.0](https://github.com/HPMLL/BurstGPT) | 1/2号覆盖连续121天，含相对时间戳、模型、输入/输出/总Token和`Log Type`；3号另覆盖110天并额外含`Session ID`、`Elapsed time`。1/2号不支持Session或耗时特征 | 核心训练、验证、测试以及跨时期稳健性分析 | 必须使用 |
| [Azure LLM Inference Dataset 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | 2024年5月10-19日，多种Azure LLM服务；含时间戳、Context Tokens、Generated Tokens | 独立小样本复现实验、检查结论能否迁移到另一真实服务 | 建议使用 |
| [Azure LLM Inference Dataset 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) | 2023年11月11日，两种Azure LLM服务；含时间戳、输入和输出Token | 补充性突发案例研究；时间太短，不适合主模型训练 | 可选 |
| [Azure LMM Inference Dataset 2025](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md) | 2024年10月采集的多模态推理轨迹，增加图像数量字段 | 可在展望中讨论多模态负载，不应与文本LLM直接合并 | 非核心 |

截至目前可核实的BurstGPT最新公开版本是2026年1月发布的v2.0。其完整文件保留Response Tokens为0的失败请求，同时提供删除失败记录的`without_fails`版本。

建议：

- 主实验使用`BurstGPT_without_fails_1/2/3.csv`，研究“成功服务的实际Token负载”。
- 完整文件只用于失败请求比例和数据质量稳健性分析。
- 不将Azure数据与BurstGPT直接拼接，因为服务规模、时间范围和匿名化方式不同。
- Azure 2024应作为独立复现实验，而不是增加训练样本。

数据质量审查直接影响了这一设计：请求级记录必须先聚合成预测时点，未来生成Token不能作为当期特征；不同数据批次也应分别验证，不能无条件混合。

---

## 4. 研究背景、研究意义与可行性说明（约500字）

**研究背景：** 随着生成式人工智能应用进入在线搜索、编程辅助和企业服务，LLM推理系统面临高度波动的请求到达率及差异显著的输入、输出Token长度。Token负载不仅决定GPU计算量、KV Cache占用和服务延迟，也影响动态扩缩容和能源消耗。BurstGPT公开了由Microsoft Azure支持的真实GPT-3.5和GPT-4服务轨迹，表明LLM请求同时具有周期性、长尾性和突发性 [1]。因此，在突发到来前预测短期Token需求，已经成为LLM服务资源管理的重要问题。

**研究意义：** 现有研究主要关注批处理、内存管理、Prefill-Decode分离和资源调度，而对真实Token负载的短期可预测性及不同预测模型在突发时段的表现关注相对有限。本研究比较XGBoost与LSTM：前者能够利用滞后、滚动统计和服务结构特征，后者能够学习连续时间依赖。研究不仅评价平均预测误差，还评价突发事件的Precision、Recall和F1，从而为GPU预留、自动扩缩容及容量规划提供更贴近实际决策的依据。

**可行性：** BurstGPT v2.0提供超过两百天、约千万级请求记录以及统一字段和开放许可，数据规模足以支持时间序列训练。请求聚合后仅约数万条5分钟观测，普通电脑或Google Colab即可完成XGBoost和单层LSTM训练。研究者已有Python基础，每周投入15-20小时，在六周内完成数据清洗、基线、两个模型、突发评价及论文写作具有较高可行性。主要风险是时间泄漏和LSTM过度调参，可通过严格时间切分、训练集拟合标准化器和早停机制控制。

---

## 5. 六周详细执行计划

本计划假定使用 Windows + PowerShell + Python；数据已放在项目的`Dataset/`目录。为避免“先写很多代码、最后发现数据或切分有问题”，每一周都以可运行脚本、可检查的中间文件和固定交付物结束。建议将所有命令在项目根目录执行；代码可使用 VS Code（安装 *Python* 与 *Jupyter* 扩展）或 PyCharm 编辑，Notebook 用于探索，`.py` 脚本用于最终可复现运行。

### 5.1 第0天：一次性完成环境与目录准备（第1周的第一个半天，约2小时）

1. **确认项目与原始数据位置。** 在 VS Code 中打开项目根目录；在 PowerShell 执行`Get-ChildItem Dataset`，确认至少存在`level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv`、`..._2_...csv`和`..._3_...csv`。主实验只读`without_fails`文件，不移动、不修改原始 CSV；`complete`文件仅用于失败请求比例检查。
2. **建立可复现的 Python 环境。** 在 PowerShell 依次执行：

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow shap scipy statsmodels jupyter joblib pyyaml pyarrow
   pip freeze > requirements.txt
   ```

   若`py -3`不可用，用`python -m venv .venv`代替。VS Code 中按`Ctrl+Shift+P`，选择 **Python: Select Interpreter**，选取`.venv\Scripts\python.exe`；新建 Notebook 时同样选择这个内核。若电脑没有可用 GPU，TensorFlow 使用 CPU 即可，聚合后的5分钟序列规模仍可训练单层 LSTM。
3. **建立目录和文件骨架。** 在 VS Code 文件树中创建如下目录；只将生成物写入`data/processed`、`models`、`outputs`，不要写回`Dataset`：

   ```text
   data/processed/        # 清洗后与聚合后的 parquet/csv
   notebooks/             # 01-EDA.ipynb 等探索记录
   src/                   # 可复现 Python 脚本
   configs/               # YAML 参数文件
   models/                # joblib 与 .keras 模型
   outputs/figures/       # 所有论文图
   outputs/tables/        # 所有 CSV/Markdown 结果表
   docs/                  # 数据字典、文献矩阵、实验日志
   ```

   创建`src/01_clean.py`、`src/02_build_series.py`、`src/03_features.py`、`src/04_baselines.py`、`src/05_xgboost.py`、`src/06_lstm.py`、`src/07_evaluate.py`和`src/utils.py`空文件；创建`configs/base.yaml`，先写入随机种子`42`、频率`5min`、预测窗口`[5, 15, 60]`和训练/验证/测试比例`[0.70, 0.15, 0.15]`。每个脚本都应支持`python src/脚本名.py`直接运行，并在开头设置`random.seed(42)`、`numpy.random.seed(42)`和`tf.keras.utils.set_random_seed(42)`。
4. **初始化版本管理与实验日志。** 在 PowerShell 执行`git status`，保留已有 Git 历史；新建`.gitignore`，至少忽略`.venv/`、`__pycache__/`、`.ipynb_checkpoints/`、`models/`、`data/processed/`和大型`outputs/`文件。新建`docs/experiment_log.md`，按“日期—脚本/配置—数据版本—结果—决定—下一步”记录每一次正式实验。原始数据太大，不应提交至 Git。
5. **设定命名与验收规则。** 图文件采用`fig_01_daily_load.png`格式，结果表采用`table_01_baseline_test.csv`格式，模型采用`xgb_h15.joblib`、`lstm_h15.keras`格式。任何一个正式结果都必须能追溯到：输入数据文件名、代码版本/提交号、配置文件、随机种子和运行时间。

[基础配置](F:\A - 科研项目\Project\LLM服务的Token负载预测与突发流量识别——基于BurstGPT真实轨迹的XGBoost与LSTM比较\configs\base.yaml)
[环境状态与 TensorFlow 补装命令](F:\A - 科研项目\Project\LLM服务的Token负载预测与突发流量识别——基于BurstGPT真实轨迹的XGBoost与LSTM比较\docs\environment_status.md)
[实验日志](F:\A - 科研项目\Project\LLM服务的Token负载预测与突发流量识别——基于BurstGPT真实轨迹的XGBoost与LSTM比较\docs\experiment_log.md)
[依赖清单](F:\A - 科研项目\Project\LLM服务的Token负载预测与突发流量识别——基于BurstGPT真实轨迹的XGBoost与LSTM比较\requirements.txt)

### 5.2 第1周：理解数据、清洗数据并冻结研究方案（15–18小时）

**目标：** 得到可验证的请求级清洗数据，明确每个变量的含义和研究中可使用的特征，不能在本周训练模型。

1. **建立文献矩阵（约3小时）。** 用 Excel、Google Sheets 或`docs/literature_matrix.md`建立表格，列为“编号、研究问题、数据、方法、指标、与本文的关系、可引用位置、局限”。先精读文献[1]、[10]、[11]、[15]，再阅读[2]–[9]；每篇至少填入两项可用于论文的结论及对应页码/章节。第1周结束前，把研究问题固定为第1.1节的三项问题，不再增加模型种类。
2. **抽样查看 CSV 和真实字段（约1小时）。** 在 Notebook 中用`pandas.read_csv(..., nrows=5)`分别读取三个`without_fails`文件，执行`df.columns.tolist()`、`df.dtypes`和`df.head().T`。将实际字段名与计划中的时间戳、输入/输出 Token、模型、日志类型、Session、调用方式逐项对应；不要依据网上字段名直接写代码。把结果写入`docs/data_dictionary.md`，包括字段名、类型、单位、缺失含义、能否作为预测时点特征、清洗规则。
3. **编写分块清洗脚本（约4小时）。** 在`src/01_clean.py`中使用`pd.read_csv(path, chunksize=200_000)`逐块读取，避免一次性占满内存。每块完成以下操作：统一列名为小写下划线；`pd.to_datetime(..., utc=True, errors="coerce")`转换时间；用`pd.to_numeric(errors="coerce")`转换 Token 列；删除无法解析时间、Token 为负数、关键字段缺失的记录；依据完整重复键或所有字段`drop_duplicates()`去重；新增`total_tokens = input_tokens + output_tokens`（实际列名按数据字典替换）。将清洗结果以`parquet`格式保存到`data/processed/requests_1.parquet`和`requests_2.parquet`，并保留每一步删除行数。
4. **做数据质量审计（约3小时）。** 在`notebooks/01_data_quality.ipynb`或脚本中计算并保存：每个文件的原始行数、清洗后行数、删除比例、各字段缺失率、重复行数、零输出 Token 比例、时间最小/最大值、模型和日志类型的频数。将汇总表保存为`outputs/tables/table_00_data_quality.csv`。使用`matplotlib`绘制请求时间直方图和`total_tokens`的对数直方图，检查是否存在明显时间断层或极端异常值。
5. **检查失败记录和数据批次边界（约2小时）。** 用相同的分块方式读取`complete_1`，只统计`output_tokens == 0`或数据说明中定义的失败条件的比例；不把它混入主训练数据。分别核对1、2、3号文件的时间范围、字段一致性与频率，形成`docs/data_dictionary.md`中的“批次使用规则”：1和2用于主实验，3只用于跨时期稳健性测试。
6. **执行小样本端到端冒烟测试（约2小时）。** 从清洗数据取连续两天，完成一次5分钟聚合并保存一个小型 CSV，验证时间、Token 总和和行数无误；这是后续建模的最低前提。用`python src/01_clean.py`从命令行重跑清洗脚本，确认输出和 Notebook 一致。

**本周验收与交付：** `docs/literature_matrix.md`、`docs/data_dictionary.md`、`src/01_clean.py`、清洗后的1/2号 Parquet、`table_00_data_quality.csv`和一条包含关键决定的实验日志。若真实字段无法支持某项特征，应立即在第1.3节和后续计划中删去该特征，而不是用虚构字段补齐。

### 5.3 第2周：构造5分钟时间序列、完成 EDA 与基线（16–20小时）

**目标：** 生成唯一的、带零值时间窗的5分钟目标序列，确定突发定义，并建立必须击败的朴素基线。

1. **合并主实验批次但保持时间顺序（约2小时）。** 在`src/02_build_series.py`读取`requests_1.parquet`和`requests_2.parquet`，用`pd.concat`合并后按时间升序排序。先使用`assert df[timestamp].is_monotonic_increasing`检查排序，再保存原始批次标识，以便后续核查是否在批次边界产生时间空洞。
2. **聚合为完整5分钟网格（约3小时）。** 将时间设为索引后执行`resample("5min")`：目标`token_load`为`total_tokens.sum()`；同时聚合`request_count`、`mean_request_tokens`、`mean_response_tokens`，以及 GPT-4/API 的请求数。1/2号没有Session字段，不计算会话特征。使用`asfreq("5min", fill_value=0)`或重建完整`date_range`，确保无请求的时间窗也有一行且负载为零；均值类特征在请求数为零时保留`NaN`，稍后只用历史信息填充。保存为`data/processed/series_5min.parquet`。
3. **人工核验聚合正确性（约1小时）。** 随机挑选三个5分钟窗口，回到请求级 Parquet，用布尔条件筛选原始请求并手动求和，与聚合结果逐项比较；将窗口起止时间和两种计算结果写入`docs/experiment_log.md`。这一检查可防止时区、闭区间和重复聚合错误。
4. **完成探索性分析（约5小时）。** 在`notebooks/02_eda.ipynb`用`matplotlib`和`seaborn`按统一风格绘制并保存：完整时间趋势、一个典型周的5分钟曲线、按相对日内时段/周内相位的平均负载热图、Token 负载对数分布、请求数与 Token 负载散点图、模型/`Log Type`占比随时间变化、ACF/PACF 图（`statsmodels.graphics.tsaplots`）。每张图下用1–2句话记录观察结论，图只展示能服务于特征、基线或讨论部分的信息；不得把相对周相位解释为真实星期或周末。
5. **只用训练期定义切分点与突发阈值（约2小时）。** 在序列建立后按时间前70%、中15%、后15%切成`train`、`valid`、`test`，打印每段的起止时间和行数，写入`outputs/tables/table_01_split_summary.csv`。仅对`train["token_load"]`计算`p95_threshold = np.quantile(..., 0.95)`；用这个固定阈值为三段分别创建`actual_burst`。严禁对验证集或测试集重新计算 P95。
6. **实现两条基线（约3小时）。** 在`src/04_baselines.py`为每个预测窗口`h ∈ {1,3,12}`个5分钟步长生成预测：Persistence 使用`y[t-h]`，Seasonal Naive 使用同一时刻前一天的`y[t-288]`。预测必须先向后对齐到未来目标，之后再计算误差，不能将未来真实值复制到当前行。生成`pred_persistence_h15`和`pred_seasonal_h15`等列，保存验证与测试预测表。
7. **先跑通评价函数（约2小时）。** 在`src/07_evaluate.py`实现 MAE、RMSE、sMAPE（分母加极小量防止0/0）、Precision、Recall、F1、Average Precision/PR-AUC 和混淆矩阵；对基线的预测负载用同一训练集 P95 转为`predicted_burst`。将基线验证/测试指标写入`outputs/tables/table_02_baseline_results.csv`，并画出一段典型日期的“真实值—两条基线—P95阈值”折线图。

**本周验收与交付：** `series_5min.parquet`、EDA Notebook、至少8张编号图、`table_01_split_summary.csv`、固定的训练集 P95 值、两条可复现基线及`table_02_baseline_results.csv`。第2周结束时若清洗、聚合、切分或基线仍不能一键复现，停止 Azure 扩展实验，先修复主流程。

### 5.4 第3周：特征工程与 XGBoost（16–20小时）

**目标：** 用只包含预测时点可得信息的特征训练 XGBoost，并保留可解释的调参与特征重要性证据。

1. **定义特征可用性清单（约1小时）。** 在`docs/data_dictionary.md`增加“时点`t`可用/不可用”一列。允许：历史负载、历史请求结构、日历时间；禁止：`t+h`的输出 Token、总 Token、请求数、未来模型占比。对每一项准备写入模型的列都能回答“在预测时点如何获得”。
2. **生成无泄漏特征（约4小时）。** 在`src/03_features.py`中先按时间排序，再使用`shift(1)`创建滞后特征`lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, lag_48, lag_288`；滚动均值、标准差、最大值必须写成`series.shift(1).rolling(window).agg(...)`，窗口取3、6、12个5分钟点（15、30、60分钟）。添加`relative_day_sin/cos`和`relative_week_sin/cos`，不添加`is_weekend`或真实星期标签；模型/API占比采用过去窗口的历史聚合值。目标按`target_h15 = token_load.shift(-3)`生成。最后删除因滞后或目标移位产生的缺失行。
3. **按目标时间严格切分（约1小时）。** 对每个预测窗口，样本所属集合由**目标时间**决定：`target_time = feature_time + h × 5分钟`；训练样本的目标不得进入验证期，验证样本的目标不得进入测试期。将特征名称、样本数、每个集合的起止时间写入`outputs/tables/table_03_feature_split_summary.csv`。这是本研究最关键的时间泄漏检查。
4. **先训练默认模型，验证数据管道（约2小时）。** 用`xgboost.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)`训练15分钟模型；只用训练集拟合，不需要标准化。将验证集预测和真实值存为`outputs/tables/pred_xgb_valid_h15.csv`，检查预测是否全为常数、负数或与真实值时间错位。
5. **小范围调参（约4小时）。** 使用`RandomizedSearchCV`配合`TimeSeriesSplit`，或明确的验证集循环，只搜索有限组合：`max_depth`（3、5、7）、`learning_rate`（0.02、0.05、0.1）、`n_estimators`（300、600、1000）、`min_child_weight`（1、5、10）、`subsample`（0.7、0.9、1.0）和`colsample_bytree`（0.7、0.9、1.0）。评分以验证集 MAE 为主，同时记录 F1；每次结果写入`outputs/tables/xgb_tuning_log.csv`。不得读取测试集或基于测试集选择参数。
6. **定型、重训与测试（约2小时）。** 选择验证 MAE 最优且模型不过度复杂的一组参数，保存到`configs/xgb_h15.yaml`。用训练+验证数据重训最终模型，然后仅一次预测测试集；使用`joblib.dump`保存为`models/xgb_h15.joblib`，保存测试预测为`pred_xgb_test_h15.csv`。5分钟、60分钟任务复用同一流程及独立配置。
7. **解释与合理性检查（约3小时）。** 先输出 XGBoost 原生`feature_importances_`；再对验证集随机抽取不超过2,000行，用`shap.TreeExplainer`计算 SHAP，绘制汇总图和前3个特征依赖图。检查最重要特征是否均为历史或日历变量；若出现未来变量，立即回退修复特征脚本并重新训练。

**本周验收与交付：** `src/03_features.py`、特征清单与时间泄漏检查表、XGBoost 三个窗口的配置和模型、调参日志、验证/测试预测 CSV、特征重要性与 SHAP 图。模型必须至少在主要15分钟任务上优于或可解释地不优于两条基线；任何“不优于”也是有效研究结果，不能在测试集继续调参。

### 5.5 第4周：序列构造与 LSTM（18–20小时）

**目标：** 使用与 XGBoost 相同的时间边界与目标定义训练一个克制、稳定的 LSTM，而不是追求复杂网络。

1. **确定 LSTM 的最小输入变量（约1小时）。** 主模型先使用过去的`token_load`、`request_count`、平均输入/输出 Token、模型/API历史占比及周期特征；为公平比较，可额外运行“仅历史负载”版本作为消融。选择12小时（144步）或24小时（288步）的回看窗口，先用12小时；在配置中固定为`lookback: 144`。
2. **先按时间切分、再拟合标准化器（约3小时）。** 对训练、验证、测试分别构造时间边界；仅用训练期的输入列拟合`sklearn.preprocessing.StandardScaler`，再分别`transform`三段数据。目标`token_load`若标准化，也只能由训练期拟合另一个 scaler。用`joblib.dump`保存两个 scaler 到`models/`，绝不在全部数据上调用`fit_transform`。
3. **构造三维滑动窗口（约3小时）。** 在`src/06_lstm.py`写`make_sequences(features, target, lookback, horizon)`函数：每条输入为`[t-lookback+1, ..., t]`，标签为`t+horizon`；返回形状`(样本数, lookback, 特征数)`。针对每段分别构造，且验证/测试可在输入窗口中携带前一段的**历史**观测，但其标签必须属于本段。打印三段形状与首尾时间，人工核对一条样本的输入终点、标签时间和目标值。
4. **建立基准网络（约2小时）。** 使用 Keras Sequential：`Input` → `LSTM(32, dropout=0.1)` → `Dense(1)`；编译为`optimizer="adam"`、`loss="mae"`、额外指标`rmse`。禁止本周新增 Transformer、Attention、CNN 或多层宽网络。执行`model.summary()`并把参数量写入实验日志。
5. **训练并防止过拟合（约3小时）。** 使用`validation_data=(X_valid, y_valid)`、`EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)`和`ModelCheckpoint("models/lstm_h15.keras", save_best_only=True)`；最大`epochs=100`、`batch_size=64`，实际以早停为准。每次训练记录配置、最佳 epoch、训练/验证 MAE。绘制并保存训练与验证损失曲线；如验证损失持续高于训练损失，先缩小网络或增加轻微 dropout，而不是不断加层。
6. **只在验证集选择一次配置（约2小时）。** 允许比较的配置仅限单层32与64单元、回看12和24小时、dropout 0与0.1；最多4次正式运行。以验证 MAE 为主、验证 F1 为辅选出配置，写入`configs/lstm_h15.yaml`。如训练不稳定或不优于季节朴素基线，冻结为单层32/64单元并继续后续比较，不扩大搜索范围。
7. **重训、逆变换和测试（约3小时）。** 使用训练+验证数据按固定最佳 epoch 重训；将模型输出用目标 scaler 逆变换回 Token 原始单位，负预测截断为0。仅预测一次测试集，输出`pred_lstm_test_h15.csv`，与 XGBoost 同样按训练集 P95 生成突发标签。依同一代码流程完成5分钟与60分钟模型。
8. **保存训练证据（约1小时）。** 保存`.keras`模型、scaler、配置 YAML、训练曲线、模型摘要、预测 CSV 和所有随机种子；重启内核后用`python src/06_lstm.py --horizon 15`完整运行一次，确认不依赖 Notebook 的隐藏变量。

**本周验收与交付：** 三个预测窗口的 LSTM 模型与配置、scaler、训练曲线、预测表、样本形状/时间边界检查记录。如 LSTM 的训练或验证流程不稳定，固定为单层32或64单元，不再扩大网络，保证第5周可完成公平对比。

### 5.6 第5周：统一评估、稳健性与误差分析（16–20小时）

**目标：** 以同一测试集、同一阈值、同一指标得到可直接写入论文的结论，并区分总体表现和突发时段表现。

1. **锁定版本，禁止继续基于测试集调参（约0.5小时）。** 在`docs/experiment_log.md`记录最终的基线、XGBoost、LSTM配置和文件名；此后测试集只用于汇报，模型选择一律以第3、4周验证结果为准。
2. **汇总统一预测长表（约2小时）。** 将各模型、各预测窗口的测试集预测整理为长格式表：`timestamp`、`horizon`、`model`、`y_true`、`y_pred`、`actual_burst`、`predicted_burst`；检查所有模型在同一`timestamp+horizon`有相同真实标签。保存`outputs/tables/test_predictions_all_models.csv`。
3. **计算主要指标和置信信息（约3小时）。** 用`src/07_evaluate.py`分别计算 MAE、RMSE、sMAPE、Precision、Recall、F1、PR-AUC、混淆矩阵 TP/FP/FN/TN，以及相对 Seasonal Naive 的 MAE 改善率。主要表按15分钟窗口排序，5/60分钟放入附表；将所有数字保留3位有效小数，保存为`table_04_main_results.csv`和`table_05_burst_results.csv`。
4. **制作核心图表（约3小时）。** 绘制并编号保存：(a) 典型测试周的真实负载与三类预测曲线；(b) 突发窗口的局部放大图及训练集 P95 线；(c) 各模型 MAE/F1 对比柱形图；(d) PR 曲线；(e) 三个模型的突发混淆矩阵。所有图使用同一单位、颜色和图例，图下在实验日志中写出一句解释，防止图表成为无结论装饰。
5. **做预先定义的误差分析（约3小时）。** 仅按研究计划中的维度分组：真实负载是否为突发、相对日内时段、相对周内相位、GPT-4占比高/低、API占比高/低；比较每组的 MAE 和偏差`y_pred-y_true`。对于总序列模型，分组用于解释对应时段的预测误差，不把未来同一时间窗的构成指标当作预测特征，也不把相对周相位命名为真实星期/周末。输出`table_06_segment_errors.csv`和2–3张图。
6. **最小消融实验（约2小时）。** 只针对15分钟 XGBoost，比较“完整特征”“去除服务结构特征”“仅滞后+日历特征”三种设置，配置与训练/验证边界不变。消融只在验证集选方案、测试集汇报最终三组，不扩展为大量模型。
7. **跨时期稳健性测试（约4小时）。** 用第1周完全相同的清洗和聚合规则处理`without_fails_3`；不与1/2号数据重新混合。将主模型直接应用于3号数据，或明确地以3号数据内部按时间切分重新训练后说明这是“跨时期复现”而非“外部零样本测试”。两种设计只能选一种并在`docs/experiment_log.md`写清楚。所有阈值、特征可用性和评估规则保持一致，输出`table_07_robustness_burstgpt3.csv`。
8. **可选 Azure 扩展（仅主流程稳定时，最多2小时）。** 读取本地 Azure 2024 CSV 的少量必要列，按相同5分钟聚合、切分、基线和主要模型运行；因字段不完全一致，删去不可获得的服务结构特征，并在论文中标注为独立复现实验。不得将它与 BurstGPT 拼接，也不得因其结果差异而修改主实验设计。

**本周验收与交付：** 锁定的主结果表、突发结果表、预测长表、至少5幅核心结果图、最小消融表、BurstGPT_3稳健性表和写明测试使用次数的实验日志。第5周结束即停止调参，把剩余时间留给结果解释、图表校对和论文写作。

### 5.7 第6周：论文、可复现性复核与展示（18–20小时）

**目标：** 交付一个他人能按 README 复跑、结论与图表可追溯的研究包，以及一版可展示的论文初稿。

1. **先完成结果叙事骨架（约2小时）。** 在论文文档中按“研究问题 → 数据与时间切分 → 模型与阈值 → 总体准确率 → 突发识别 → 误差/稳健性 → 局限与意义”建立标题。先把第5周已锁定的表格和图表编号插入，给每一项写一句结论，避免先写长篇背景后发现结果无法支撑论点。
2. **撰写方法部分（约3小时）。** 明确写出：请求级到5分钟聚合公式、`total_tokens`定义、主实验数据批次、清洗规则、预测窗口、以目标时间为准的70/15/15切分、训练集 P95 阈值、每个模型输入、调参仅用验证集、全部指标公式或引用。特别检查每一处“未来信息不可用”和“仅训练集拟合”是否与代码一致。
3. **撰写结果与讨论（约3小时）。** 每张核心表/图遵循“观察到什么—量化证据—可能原因—不应过度推断什么”的四句结构。总体 MAE 最优与突发 F1 最优不一致时，作为研究发现分别讨论；若 LSTM 不优于 XGBoost，如实解释数据量、特征工程、短期任务或网络容量等可能原因，不将其表述为训练失败。
4. **完成引言、相关工作和结论（约3小时）。** 使用第1周文献矩阵核对每项引用：引言说明问题与贡献，相关工作按“工作负载/服务系统”“预测方法”“研究缺口”组织，结论只重述已被实验支持的发现。用 Zotero、EndNote 或 Word 引文管理器统一参考文献格式，逐个打开 DOI/链接检查可访问性。
5. **编写可复现 README（约3小时）。** 在根目录`README.md`（若已有则补充）写清：项目目的、硬件/软件要求、Python版本、环境安装、原始数据应放置的位置、按顺序运行的命令、每一步产物位置、预计运行时间、随机种子、如何生成论文表图、数据许可/不提交原始数据的说明。例如给出：

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python src/01_clean.py
   python src/02_build_series.py
   python src/04_baselines.py
   python src/05_xgboost.py --horizon 15
   python src/06_lstm.py --horizon 15
   python src/07_evaluate.py --horizon 15
   ```

6. **从干净状态复跑关键流程（约3小时）。** 新开 PowerShell 和 Python 内核，删除或移动**仅生成的**`data/processed`、`models`和`outputs`副本到备份位置后，严格按 README 从`01_clean.py`重跑至`07_evaluate.py`。逐项比较重跑表格与已写入论文的结果；如果不同，先检查随机种子、数据版本、包版本和隐式 Notebook 状态。不要删除`Dataset/`中的原始 CSV。
7. **进行最终质量检查（约1小时）。** 用清单逐项确认：所有图轴有单位；表格有数据集和预测窗口说明；百分比/Token 单位一致；无测试集调参；P95来源于训练集；所有文件名与 README 一致；`git status`中没有意外的大型文件或密钥。为当前代码和文档创建一次有意义的 Git 提交。
8. **准备15分钟展示（约2小时）。** 用 PowerPoint、Google Slides 或 Canva 制作8–10页：问题与价值、数据、无泄漏流程、模型、主要结果、突发案例、稳健性/局限、结论与下一步。每页只保留一个主要信息，优先复用论文中已编号且经核对的图；演练一次并将讲稿控制在15分钟以内。

**本周验收与交付：** 论文初稿、最终图表和表格、完整 README、`requirements.txt`、可从干净环境复跑的代码、实验日志和15分钟展示材料。

### 5.8 六周检查点与止损规则

| 检查点 | 必须回答的问题 | 未通过时的处理 |
|---|---|---|
| 第1周末 | 原始字段、清洗规则和主实验批次是否已写入数据字典，并能从原始 CSV 重建清洗数据？ | 暂停 EDA 与模型工作，优先修复字段映射、时间格式或清洗脚本。 |
| 第2周末 | 5分钟序列、目标时间切分、训练集 P95 与两条基线能否一键重跑？ | 取消 Azure 扩展，不进入特征调参，先完成端到端基线。 |
| 第3周末 | XGBoost 特征是否全部在预测时点可得，且验证集结果已保存？ | 删除疑似泄漏特征；不可使用测试集挑选特征或参数。 |
| 第4周末 | LSTM 是否能稳定训练、保存和复跑？ | 固定为单层32或64单元，停止扩大网络与搜索空间。 |
| 第5周末 | 主结果、突发结果和稳健性分析是否已锁定？ | 停止所有调参；仅修复代码错误或图表/结果不一致。 |
| 第6周末 | README 是否可从干净环境复跑，论文结论是否逐一有表图支撑？ | 优先完成复现性与结果核对，删去没有证据支持的扩展结论。 |

---

## 6. Introduction（约1000字，含8篇文献综述）

### Introduction

生成式人工智能正在由离线模型训练迅速转向面向大量用户的在线推理服务。与传统网络服务相比，大语言模型请求不仅具有到达时间上的不确定性，而且每项请求的输入与输出Token长度差异显著。Token数量直接影响Prefill计算量、Decode持续时间、KV Cache占用以及GPU资源需求。当大量长请求在短时间内集中到达时，服务系统可能出现排队增长、延迟上升和服务等级目标违约。因此，仅预测请求数量不足以完整描述LLM服务压力，对短期Token负载进行预测并提前识别突发流量，具有更直接的资源管理价值。

云计算领域已经对生产工作负载预测进行了长期研究。Resource Central通过分析大规模Azure虚拟机轨迹，说明工作负载历史、资源类型和生命周期信息能够支持更有效的资源配置决策 [2]。Shahrad等人对Azure Functions生产轨迹的研究进一步揭示了请求频率具有明显的异质性、长尾性和时间波动，并指出服务系统必须针对突发调用改进资源管理策略 [3]。然而，虚拟机或Serverless函数的资源指标不能直接表示LLM推理负载，因为后者还受到输入Token、生成Token和模型类型的共同影响。

现有LLM服务研究主要从系统优化角度应对这种不确定性。Orca提出迭代级调度与选择性批处理，使不同生成长度的请求能够更有效地共同执行 [4]。vLLM通过PagedAttention降低KV Cache碎片和重复占用，显著提高了LLM服务吞吐率 [5]。这些研究说明请求长度和Token生成过程会直接影响系统效率，但其重点是给定工作负载后的执行优化，而不是提前预测未来负载。

部分研究开始直接关注流量突发与资源配置。AlpaServe利用模型并行与统计复用处理生产环境中的波动请求，并表明合理的模型部署策略能够提升系统可承受的突发程度 [6]。Splitwise进一步指出LLM推理的Prefill和Decode阶段分别呈现计算密集与内存密集特征，因而输入Token和输出Token不应被视为完全相同的资源需求 [7]。DynamoLLM则根据负载变化动态调整LLM推理集群配置，在性能、成本与能源效率之间进行权衡 [9]。这些成果说明，准确获得未来短期Token需求可能为批处理、GPU分配和自动扩缩容提供有价值的先验信息。

真实生产数据的缺乏曾限制相关预测研究。BurstGPT发布了由Microsoft Azure支持的ChatGPT和GPT-4服务轨迹，包含请求相对时间、模型类型、输入与输出Token及API/Conversation日志类型；公开批次的字段并不完全一致，1/2号没有Session或响应耗时字段，只有3号额外提供这些字段 [1]。该数据揭示了LLM流量的日周期、周周期、长尾分布与突发现象。不过，现有代表性研究更多利用此类轨迹开展工作负载刻画或系统回放，对不同预测模型能否提前识别Token负载突发，以及模型在普通时段与峰值时段是否具有不同优势，仍缺少系统比较。

基于此，本文利用BurstGPT真实轨迹构建5分钟粒度的Token负载时间序列，并比较XGBoost与LSTM在未来5分钟、15分钟和60分钟预测任务上的表现。XGBoost通过滞后负载、滚动统计、周期信息、模型构成和调用方式等人工特征建立非线性映射；LSTM则从连续历史序列中学习短期与周期性依赖。本文将15分钟作为主要预测窗口，并将超过训练集第95百分位的未来负载定义为突发事件。除MAE、RMSE和sMAPE外，研究还使用Precision、Recall、F1和PR-AUC评价模型对突发事件的识别能力。

本文拟回答第1.1节冻结的三个问题：第一，5分钟Token负载的时间变化、分布与突发特征；第二，XGBoost与LSTM在5、15、60分钟预测中相对两条朴素基线的误差表现；第三，预测负载识别训练集P95突发的能力及其在3号跨时期批次上的稳健性。研究贡献在于建立一套无时间泄漏且可复现的LLM负载预测流程，将连续负载预测与突发识别纳入统一评价框架，并通过跨时期测试检验结论的稳定性。研究结果可为LLM服务的GPU预留、动态扩缩容和容量规划提供实证依据，同时展示传统机器学习与深度学习模型在真实系统数据上的适用条件和局限性。
