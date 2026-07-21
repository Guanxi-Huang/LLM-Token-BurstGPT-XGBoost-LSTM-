# 数据与目录约定

## 已确认的本地原始数据

`Dataset/`中已存在以下主实验文件，项目脚本只读这些原始 CSV：

- `level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv`
- `level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv`
- `level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv`

其中1、2号文件用于主实验，3号文件仅用于第5周跨时期稳健性测试。清洗后的 Parquet/CSV 一律写入`data/processed/`；模型写入`models/`；图和表分别写入`outputs/figures/`与`outputs/tables/`。

## 命名规则

- 图：`fig_01_daily_load.png`
- 表：`table_01_baseline_test.csv`
- XGBoost 模型：`xgb_h15.joblib`
- LSTM 模型：`lstm_h15.keras`
- 预测结果：`pred_<model>_<split>_h15.csv`

所有正式结果都应能追溯到原始数据文件、配置文件、随机种子、运行命令和实验日志记录。
