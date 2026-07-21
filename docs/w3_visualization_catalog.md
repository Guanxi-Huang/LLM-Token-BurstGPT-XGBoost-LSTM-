# W3结果可视化图册：Python＋RStudio

本图册把W3的无泄漏特征、目标时间切分、有限调参、默认/定型模型、一次性测试、预测诊断和解释性证据整理为两套可复现静态图。Python首先从正式W3表格与预测CSV导出共享绘图数据；RStudio只读取这些CSV，因此两套图使用相同样本、指标、选中参数和代表性测试日。

## 图形契约与索引

| 编号 | 分析问题与预期结论 | 图形家族/形式 | 数据充分性 | Python图 | RStudio图 |
|---|---|---|---|---|---|
| W3-01 | 三个窗口是否按目标时间严格分段？目标区间在train/valid/test边界不交叉。 | 趋势/目标时间区间图 | 3窗口×3分段，共9条区间；足以逐项核对边界。 | `fig_w3_py_01_target_time_splits.png` | `fig_w3_r_01_target_time_splits.png` |
| W3-02 | 36个入模列来自哪些可用信息？全部属于历史负载、历史请求结构或相对周期。 | 比较/水平条形图 | 36个逐列审计字段，按3类汇总。 | `fig_w3_py_02_feature_availability.png` | `fig_w3_r_02_feature_availability.png` |
| W3-03 | 有限调参中的MAE—F1权衡如何？每个窗口的选中点由最低验证MAE决定。 | 关系/分面散点图 | 每窗口12个候选，共36点；达到散点图最低信息量。 | `fig_w3_py_03_tuning_tradeoff.png` | `fig_w3_r_03_tuning_tradeoff.png` |
| W3-04 | 定型XGBoost在验证集相对两条基线如何？短期窗口更接近Persistence，15分钟默认模型另作参照。 | 比较/分组条形＋默认模型标记 | 3窗口×3正式模型＋1个h15默认模型。 | `fig_w3_py_04_validation_mae.png` | `fig_w3_r_04_validation_mae.png` |
| W3-05 | 一次性测试中三类模型如何排序？15分钟XGBoost未优于两条基线。 | 比较/分组条形图 | 3窗口×3模型，共9个可比MAE。 | `fig_w3_py_05_test_mae.png` | `fig_w3_r_05_test_mae.png` |
| W3-06 | 代表性测试相对日内，预测是否跟随真实负载？Persistence对低负载状态适应更快。 | 趋势/多序列折线图 | 1个完整相对日、288个5分钟点。 | `fig_w3_py_06_h15_representative_day.png` | `fig_w3_r_06_h15_representative_day.png` |
| W3-07 | 验证到测试的尺度漂移和模型偏差多大？测试实际均值下降74.86%，XGBoost存在正偏差。 | 比较/均值哑铃图 | 验证XGBoost及测试三模型，共4组实际—预测均值。 | `fig_w3_py_07_distribution_shift.png` | `fig_w3_r_07_distribution_shift.png` |
| W3-08 | 预测合理性检查发现多少负值？所有序列非恒定且时间对齐，但原始负预测不可忽略。 | 比较/负预测比例条形图 | 7个默认/定型验证/最终测试诊断记录。 | `fig_w3_py_08_prediction_diagnostics.png` | `fig_w3_r_08_prediction_diagnostics.png` |
| W3-09 | 原生重要性与SHAP在三个窗口是否都指向历史变量？主要证据集中于短滞后、滚动负载及历史API结构。 | 矩阵/双热图 | 36特征×3窗口的两类重要性；显示综合排名前12。 | `fig_w3_py_09_importance_matrix.png` | `fig_w3_r_09_importance_matrix.png` |

## 视觉与输出约束

- 静态输出面向Word文稿，白底、深灰文字、安静网格；主色为蓝，比较色为金，必要时以橙色区分第三模型。
- 同一模型在所有图中保持一致编码：XGBoost为蓝，Persistence为深灰，Seasonal Naive为金；选中调参点使用金色实心和黑色描边。
- 颜色不是唯一编码：模型比较同时使用固定顺序、直接数值标签、线型或点形；选中候选另有描边和文字标记。
- Python输出目录：`outputs/figures/w3_python/`；RStudio输出目录：`outputs/figures/w3_rstudio/`；共享绘图表：`outputs/tables/w3_visualization/`。
- Word正文选取互补核心图，并保留两套图册与总览图作为附加复现证据。
