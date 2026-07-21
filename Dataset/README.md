# 数据集来源与命名说明

文件名使用 `level_数据集名称_时间_来源`：数字越大表示与本研究的相关性和建议使用优先级越高。

- `level_5`：BurstGPT v2.0 的 `without_fails` 轨迹，是主实验训练、验证和测试数据。
- `level_4`：BurstGPT v2.0 完整轨迹，保留 `Response tokens = 0` 的失败请求，用于失败比例和稳健性分析。
- `level_3`：Azure LLM 2024，作为独立复现实验。
- `level_2`：Azure LLM 2023，适合作为短时突发案例。
- `level_1`：Azure LMM 2025 文档所述的多模态轨迹；其实际采集期为 2024-10-15 至 2024-10-22，不与纯文本 LLM 主实验混合。

BurstGPT 的官方发布只给出连续天数和相对时间戳，未公开采集的日历年份，因此文件名中的时间字段如实使用 `Undisclosed`。

Azure LLM 2024 的官方文档仍列出两个 CSV 下载链接；截至 2026-07-18，该 Azure Blob 存储端点返回 `PublicAccessNotPermitted`，因此这两个 `level_3` 文件未能下载，未以占位文件替代。

来源：BurstGPT [Release v2.0](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)；Azure [2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)、[2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)、[2025](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md)。Azure 数据以 CC-BY 许可发布；BurstGPT 仓库以 CC-BY-4.0 许可发布。
