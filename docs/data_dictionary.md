# BurstGPT v2.0 数据字典与批次规则

## 1. 核验范围与时间语义

本字典基于本地三份 `level_5_BurstGPT_v2.0_without_fails_*.csv` 的真实表头与数据，而不是网上示例。`notebooks/01_data_quality.ipynb` 对每份文件分别执行了：

```python
df = pandas.read_csv(path, nrows=5)
df.columns.tolist()
df.dtypes
df.head().T
```

1/2号的实际列为 `Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type`；3号在此基础上增加 `Session ID, Elapsed time`。1号抽样时 `Timestamp` 推断为 `int64`，2/3号为 `float64`；Token列均为 `int64`，文本列为字符串，3号的 `Elapsed time` 为 `int64`。

`Timestamp` 的单位是从轨迹起点累计的相对秒数，公开资料没有给出采集日历年份或起始星期。清洗脚本用
`pd.to_datetime(..., unit="s", origin="unix", utc=True, errors="coerce")`
生成可供 Pandas 重采样的 `timestamp`，但这只是锚定在1970-01-01的**合成UTC时间轴**；原值保存在 `timestamp_relative_seconds`。论文不得把1970日期写成采集日期，也不得由此构造真实星期、周末、节假日等特征。

## 2. 概念字段对应

| 计划概念 | 真实字段 | 结论 |
|---|---|---|
| 时间戳 | `Timestamp` | 有；相对秒数，不是公开的日历时间。 |
| 输入Token | `Request tokens` | 有；清洗后为 `request_tokens`。 |
| 输出Token | `Response tokens` | 有；清洗后为 `response_tokens`。论文将零值按官方定义视为失败。 |
| 模型 | `Model` | 有；观察值为 `ChatGPT`、`GPT-4`。 |
| 日志类型 | `Log Type` | 有；观察值为 `API log`、`Conversation log`。 |
| Session | `Session ID` | 1/2号完全没有；3号仅Conversation记录有，所有API记录均缺失。主实验删除Session特征。 |
| 调用方式 | 无独立字段 | `Log Type`已编码API/Conversation调用模式；不得再虚构另一列。 |
| 响应耗时 | `Elapsed time` | 只在3号出现，且为请求完成后的结果；主实验不可用并有时间泄漏风险。 |

## 3. 字段级数据字典

| 来源字段 | 清洗后字段 | 批次 | 类型、单位或取值 | 缺失含义 | 时点 `t` 可用/不可用 | 清洗与使用规则 |
|---|---|---|---|---|---|---|
| `Timestamp` | `timestamp_relative_seconds`; `timestamp` | 1/2/3 | 数值，相对秒；另存UTC型合成时间 | 无法定位请求时间；属关键缺失 | 当前相对时间与历史周期相位可用；真实日历不可用 | `pd.to_numeric(errors="coerce")`后按秒转UTC；无法转换即删除；保留原相对秒。 |
| `Model` | `model` | 1/2/3 | 字符串：`ChatGPT`、`GPT-4` | 无法计算服务构成；属关键缺失 | 只能使用预测时点以前窗口的历史占比；未来请求模型未知 | 去首尾空格；空值删除；不把未来窗口模型占比作为特征。 |
| `Request tokens` | `request_tokens` | 1/2/3 | 非负整数，Token | 无法定义负载；属关键缺失 | 提前预测未来请求时未知；仅历史聚合可用 | `pd.to_numeric(errors="coerce")`；非数值或负数删除。 |
| `Response tokens` | `response_tokens` | 1/2/3 | 非负整数，Token | 无法定义负载；属关键缺失 | 请求完成后才知道；未来值仅作目标，历史聚合可用 | `pd.to_numeric(errors="coerce")`；非数值或负数删除；完整文件中0定义为失败。 |
| `Total tokens` | `reported_total_tokens` | 1/2/3 | 非负整数，Token | 来源合计不可核对，但可由两项重算 | 未来不可用；只作来源一致性审计 | 数值化后保留为核对列，不直接作为最终合计。 |
| 派生 | `total_tokens` | 1/2/3 | `request_tokens + response_tokens`，Token | 两个组成字段任一缺失时不生成 | 未来值是预测目标；仅历史滞后可作特征 | 每条记录重新计算；审计发现与来源总Token不一致为0行。 |
| `Log Type` | `log_type` | 1/2/3 | `API log`、`Conversation log` | 无法计算调用模式；属关键缺失 | 只能使用历史窗口占比；未来调用模式未知 | 去首尾空格；空值删除；同时承担“日志类型/调用方式”映射。 |
| `Session ID` | `session_id` | 仅3 | 字符串标识符 | API记录结构性缺失；Conversation记录不缺 | 主实验不可用 | 1/2号不存在；3号缺失4,724,297行（95.3237%），恰为全部API记录；不进入主模型。 |
| `Elapsed time` | `elapsed_time` | 仅3 | 数值；公开CSV未给出足以在本文确认的单位 | 缺失表示来源未提供 | 请求完成后才产生，预测时不可用 | 数值化仅用于审计；不进入主模型，也不跨批次补齐。 |
| 派生 | `source_batch` | 清洗产物 | `1`或`2`（3号本周只审计） | 不适用 | 可用于边界核查，不作为一般化预测特征 | 由文件名写入，防止拼接后丢失批次来源。 |

所有来源字段先统一为小写下划线。每个20万行块先调用 `drop_duplicates()` 等价的全字段判断；脚本另用稳定行哈希在块间延续已见键，因而也会删除跨块完全相同记录。由于1/2号没有请求唯一ID，同一秒、同模型、同Token、同日志类型的记录既可能是重复写入，也可能是不同的并发请求；按本周验收规则仍删除，但这会带来低估高并发负载的风险，论文须将其列为局限并建议后续做“保留/删除完全同值行”的敏感性分析。

### 3.1 XGBoost 入模特征的时点可用性清单（冻结）

预测原点 `t` 是最新一个已经结束的5分钟窗之后的时刻。所有历史统计均先执行 `shift(1)`；因此最晚只使用 `t-1` 窗及更早的信息。下表逐列对应 `src/03_features.py` 实际写入模型矩阵的字段，机器可读版本见 `outputs/tables/table_03_feature_availability.csv`。

| 入模列 | 时点 `t` 可用/不可用 | 在预测时点如何获得 |
|---|---|---|
| `lag_1` | 可用 | 读取 `t` 前5分钟已完成窗的 `token_load`。 |
| `lag_2` | 可用 | 读取 `t` 前10分钟已完成窗的 `token_load`。 |
| `lag_3` | 可用 | 读取 `t` 前15分钟已完成窗的 `token_load`。 |
| `lag_6` | 可用 | 读取 `t` 前30分钟已完成窗的 `token_load`。 |
| `lag_12` | 可用 | 读取 `t` 前60分钟已完成窗的 `token_load`。 |
| `lag_24` | 可用 | 读取 `t` 前120分钟已完成窗的 `token_load`。 |
| `lag_48` | 可用 | 读取 `t` 前240分钟已完成窗的 `token_load`。 |
| `lag_288` | 可用 | 读取 `t` 前24小时已完成窗的 `token_load`。 |
| `rolling_mean_3` | 可用 | `token_load.shift(1).rolling(3).mean()`，即过去15分钟。 |
| `rolling_std_3` | 可用 | `token_load.shift(1).rolling(3).std()`，即过去15分钟。 |
| `rolling_max_3` | 可用 | `token_load.shift(1).rolling(3).max()`，即过去15分钟。 |
| `rolling_mean_6` | 可用 | `token_load.shift(1).rolling(6).mean()`，即过去30分钟。 |
| `rolling_std_6` | 可用 | `token_load.shift(1).rolling(6).std()`，即过去30分钟。 |
| `rolling_max_6` | 可用 | `token_load.shift(1).rolling(6).max()`，即过去30分钟。 |
| `rolling_mean_12` | 可用 | `token_load.shift(1).rolling(12).mean()`，即过去60分钟。 |
| `rolling_std_12` | 可用 | `token_load.shift(1).rolling(12).std()`，即过去60分钟。 |
| `rolling_max_12` | 可用 | `token_load.shift(1).rolling(12).max()`，即过去60分钟。 |
| `request_count_rolling_mean_3` | 可用 | 对 `shift(1)` 后的请求数取过去3窗均值。 |
| `request_count_rolling_mean_6` | 可用 | 对 `shift(1)` 后的请求数取过去6窗均值。 |
| `request_count_rolling_mean_12` | 可用 | 对 `shift(1)` 后的请求数取过去12窗均值。 |
| `gpt4_share_history_3` | 可用 | 过去3个已完成窗的 GPT-4 请求数之和/总请求数之和。 |
| `gpt4_share_history_6` | 可用 | 过去6个已完成窗的 GPT-4 请求数之和/总请求数之和。 |
| `gpt4_share_history_12` | 可用 | 过去12个已完成窗的 GPT-4 请求数之和/总请求数之和。 |
| `api_share_history_3` | 可用 | 过去3个已完成窗的 API 请求数之和/总请求数之和。 |
| `api_share_history_6` | 可用 | 过去6个已完成窗的 API 请求数之和/总请求数之和。 |
| `api_share_history_12` | 可用 | 过去12个已完成窗的 API 请求数之和/总请求数之和。 |
| `mean_request_tokens_history_3` | 可用 | 过去3个已完成窗的输入Token总和/请求总数。 |
| `mean_request_tokens_history_6` | 可用 | 过去6个已完成窗的输入Token总和/请求总数。 |
| `mean_request_tokens_history_12` | 可用 | 过去12个已完成窗的输入Token总和/请求总数。 |
| `mean_response_tokens_history_3` | 可用 | 过去3个已完成窗的输出Token总和/请求总数。 |
| `mean_response_tokens_history_6` | 可用 | 过去6个已完成窗的输出Token总和/请求总数。 |
| `mean_response_tokens_history_12` | 可用 | 过去12个已完成窗的输出Token总和/请求总数。 |
| `relative_day_sin` | 可用 | 由轨迹起点以来的相对秒数在24小时周期中的正弦相位确定。 |
| `relative_day_cos` | 可用 | 由轨迹起点以来的相对秒数在24小时周期中的余弦相位确定。 |
| `relative_week_sin` | 可用 | 由轨迹起点以来的相对秒数在7日周期中的正弦相位确定。 |
| `relative_week_cos` | 可用 | 由轨迹起点以来的相对秒数在7日周期中的余弦相位确定。 |

明确禁止写入 `X` 的字段如下。它们可以是预测目标或事后审计字段，但不能伪装成解释变量。

| 禁止字段 | 时点 `t` 可用/不可用 | 处理规则 |
|---|---|---|
| `target_h5`、`target_h15`、`target_h60`及任何 `t+h` 输出/总Token | 不可用 | 只作为 `y`；分别由 `token_load.shift(-1/-3/-12)` 生成。 |
| `t+h` 的 `request_count` | 不可用 | 不进入特征；只能使用 `shift(1)` 后的历史聚合。 |
| `t+h` 的模型/API/Conversation占比 | 不可用 | 不进入特征；只允许过去窗口计数聚合出的历史占比。 |
| `is_weekend`、真实星期、真实日期、节假日 | 不可用 | 公开轨迹没有真实日历锚点，不创建这些字段。 |
| `Session ID`、`Elapsed time`、`source_batch` | 不可用 | 分别因结构缺失、事后产生或仅供边界审计而排除。 |

## 4. 全量质量审计结果

详细机器可读结果见 `outputs/tables/table_00_data_quality.csv`，顺序删除账本见 `outputs/tables/table_00_cleaning_steps.csv`。

| 文件 | 原始行 | 清洗后行 | 删除率 | 删除原因 | 相对秒范围 | 平均原始请求/小时 | 最大相邻间隔 | >1小时间隔数 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| without_fails_1 | 1,404,294 | 1,363,384 | 2.9132% | 完全同值行40,910；其余规则均0 | 5–5,269,973（约61天） | 959.30 | 15,240秒 | 49 |
| without_fails_2 | 3,784,213 | 3,695,526 | 2.3436% | 完全同值行88,687；其余规则均0 | 5,270,414–10,454,395（约60天） | 2,627.94 | 37,133秒 | 60 |
| without_fails_3 | 4,956,058 | 4,805,535 | 3.0372% | 完全同值行150,523；其余规则均0 | 19,440,110–28,943,983（约110天） | 1,877.32 | 18,872秒 | 57 |

- 三个 `without_fails` 文件在所有实际来源字段上的非结构性缺失率均为0；零输出Token比例均为0；负Token、不可解析时间、来源总Token不一致均为0。
- 文件内时间顺序没有倒退，但存在多处超过1小时的请求空窗，最长约4.23、10.31、5.24小时。仅凭请求轨迹无法区分真实零流量与采集断层；建模时应保留“原始覆盖缺口”标志并做敏感性检查，不能无说明地把所有长空窗都解释为真实零需求。
- 1号末尾到2号开头仅相隔441秒（7.35分钟），两者适合保持批次标记后按时间拼接。2号末尾到3号开头相隔8,985,715秒（约104天），不能拼入同一训练序列。
- `complete_1`共有1,429,737行，其中25,443行满足 `response_tokens == 0`，失败率为1.7796%。去掉这些行恰得到without_fails_1的1,404,294行；失败定义来自BurstGPT论文第1节（PDF第2页）。完整文件不进入训练。

### 图形与极端值检查

- `outputs/figures/fig_00_request_time_histogram.png` 显示请求强度明显非平稳；2号批次约相对第 90–110 天处存在长时间低活跃区，且三个文件分别有 49、60、57 个超过 1 小时的相邻请求空窗。仅凭请求轨迹无法判定这些空窗是实际零流量还是采集断层，后续聚合必须保留 coverage 标记并做排除长空窗的敏感性分析。
- `outputs/figures/fig_00_total_tokens_log_histogram.png` 显示主实验两批次均为明显长尾且分布不同。清洗后单请求最大 `total_tokens`：1号 30,306，2号 31,924；3号达到 125,607（其中最大输入 125,591），是跨时期稳健性测试必须单独报告的极端分布漂移。
- 这些最大值通过非负与合计一致性校验，但数据说明没有给出可据以删除的协议上限，因此不擅自截断或 winsorize；建模阶段应同时报告原尺度与稳健变换/截尾敏感性结果。

## 5. 模型与日志类型频数

| 批次 | ChatGPT | GPT-4 | API log | Conversation log |
|---|---:|---:|---:|---:|
| 1 | 1,189,167（84.6808%） | 215,127（15.3192%） | 1,258,587（89.6242%） | 145,707（10.3758%） |
| 2 | 3,696,864（97.6918%） | 87,349（2.3082%） | 3,704,768（97.9006%） | 79,445（2.0994%） |
| 3 | 4,201,522（84.7755%） | 754,536（15.2245%） | 4,724,297（95.3237%） | 231,761（4.6763%） |

批次间模型和日志构成差异很大，尤其2号的ChatGPT/API占比明显更高。这是分布变化而不是字段错误，可能影响跨批次预测；必须保留批次标识并报告分批表现。

## 6. 批次使用规则（冻结）

1. **1号和2号：主实验。** 清洗为 `data/processed/requests_1.parquet` 与 `requests_2.parquet`，保持原相对时间顺序和 `source_batch`；用于训练、验证、测试。
2. **3号：只作跨时期稳健性测试。** 本周核对其完整字段、时间范围、频率和清洗后行数，但不与1/2号拼接，也不使用其Session/Elapsed time扩展主模型特征。
3. **complete_1：只作失败审计。** 仅报告零响应Token比例，不混入成功请求主训练数据。
4. **字段冻结。** 主模型不使用Session、Elapsed time、真实星期/周末/节假日，也不创建独立“调用方式”字段；只允许使用预测时点以前的负载、请求结构、模型/日志类型历史占比和相对周期相位。

## 7. 两天端到端冒烟测试

从1号清洗数据的相对秒 `[0, 172800)` 构造连续两天、576个完整5分钟窗，保存为 `data/processed/smoke_2days_5min.csv`。断言结果：请求行数6,107；输入Token 4,097,782；输出Token 1,597,098；总Token 5,694,880，且每窗与全样本均满足“输入+输出=总Token”，5分钟窗请求数之和与请求级行数一致。
