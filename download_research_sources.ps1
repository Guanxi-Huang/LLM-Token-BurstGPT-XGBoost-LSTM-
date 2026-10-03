param(
    [string]$Only = ''
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$literatureDir = Join-Path $projectRoot 'references'
$datasetDir = Join-Path $projectRoot 'Dataset'
New-Item -ItemType Directory -Force -Path $literatureDir, $datasetDir | Out-Null

function Download-Source {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if ((Test-Path -LiteralPath $Destination) -and ((Get-Item -LiteralPath $Destination).Length -gt 1024)) {
        Write-Host "Skip (already present): $(Split-Path -Leaf $Destination)"
        return
    }

    $temporary = "$Destination.part"
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Force
    }

    Write-Host "Download: $(Split-Path -Leaf $Destination)"
    & curl.exe --fail --location --retry 3 --retry-delay 3 --connect-timeout 30 --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0 Safari/537.36' --output $temporary $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed ($LASTEXITCODE): $Url"
    }
    if (-not (Test-Path -LiteralPath $temporary) -or ((Get-Item -LiteralPath $temporary).Length -le 1024)) {
        throw "Downloaded file is unexpectedly small: $Url"
    }

    Move-Item -LiteralPath $temporary -Destination $Destination -Force
}

$literature = @(
    @{ Url = 'https://arxiv.org/pdf/2401.17644'; File = '01_BurstGPT_A_Real-World_Workload_Dataset_to_Optimize_LLM_Serving_Systems.pdf' },
    @{ Url = 'https://www.microsoft.com/en-us/research/wp-content/uploads/2017/10/Resource-Central-SOSP17.pdf'; File = '02_Resource_Central_Understanding_and_Predicting_Workloads_for_Improved_Resource_Management.pdf' },
    @{ Url = 'https://www.usenix.org/system/files/atc20-shahrad.pdf'; File = '03_Serverless_in_the_Wild_Characterizing_and_Optimizing_the_Serverless_Workload.pdf' },
    @{ Url = 'https://www.usenix.org/system/files/osdi22-yu.pdf'; File = '04_Orca_A_Distributed_Serving_System_for_Transformer-Based_Generative_Models.pdf' },
    @{ Url = 'https://arxiv.org/pdf/2309.06180'; File = '05_Efficient_Memory_Management_for_Large_Language_Model_Serving_with_PagedAttention.pdf' },
    @{ Url = 'https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf'; File = '06_AlpaServe_Statistical_Multiplexing_with_Model_Parallelism_for_Deep_Learning_Serving.pdf' },
    @{ Url = 'https://arxiv.org/pdf/2311.18677'; File = '07_Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting.pdf' },
    @{ Url = 'https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf'; File = '08_DistServe_Disaggregating_Prefill_and_Decoding_for_Goodput-optimized_LLM_Serving.pdf' },
    @{ Url = 'https://arxiv.org/pdf/2408.00741'; File = '09_DynamoLLM_Designing_LLM_Inference_Clusters_for_Performance_and_Energy_Efficiency.pdf' },
    @{ Url = 'https://arxiv.org/pdf/1603.02754'; File = '10_XGBoost_A_Scalable_Tree_Boosting_System.pdf' },
    @{ Url = 'https://people.idsia.ch/~juergen/lstm1997-2024head.pdf'; File = '11_Long_Short-Term_Memory.pdf' },
    @{ Url = 'https://arxiv.org/pdf/1704.04110'; File = '12_DeepAR_Probabilistic_Forecasting_with_Autoregressive_Recurrent_Networks.pdf' },
    @{ Url = 'https://arxiv.org/pdf/1912.09363'; File = '13_Temporal_Fusion_Transformers_for_Interpretable_Multi-horizon_Time_Series_Forecasting.pdf' },
    @{ Url = 'https://arxiv.org/pdf/1905.10437'; File = '14_N-BEATS_Neural_Basis_Expansion_Analysis_for_Interpretable_Time_Series_Forecasting.pdf' },
    @{ Url = 'https://www.researchgate.net/profile/Spyros_Makridakis/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward/links/5b2c9aa4aca2720785d66b5e/The-M4-Competition-Results-findings-conclusion-and-way-forward.pdf?origin=publication_detail'; File = '15_The_M4_Competition_Results_Findings_Conclusion_and_Way_Forward.pdf' }
)

$datasets = @(
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv'; File = 'level_5_BurstGPT_v2.0_without_fails_1_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_2.csv'; File = 'level_5_BurstGPT_v2.0_without_fails_2_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_3.csv'; File = 'level_5_BurstGPT_v2.0_without_fails_3_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_1.csv'; File = 'level_4_BurstGPT_v2.0_complete_1_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_2.csv'; File = 'level_4_BurstGPT_v2.0_complete_2_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv'; File = 'level_4_BurstGPT_v2.0_complete_3_Undisclosed_GitHub.csv' },
    @{ Url = 'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv'; File = 'AzureLLMInferenceTrace_code_1week.csv' },
    @{ Url = 'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv'; File = 'AzureLLMInferenceTrace_conv_1week.csv' },
    @{ Url = 'https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv'; File = 'level_2_AzureLLMInferenceTrace_code_2023-11-11_Azure.csv' },
    @{ Url = 'https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv'; File = 'level_2_AzureLLMInferenceTrace_conversation_2023-11-11_Azure.csv' },
    @{ Url = 'https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLMMInferenceTrace_multimodal.csv.gz'; File = 'level_1_AzureLMMInferenceTrace_multimodal_2024-10-15_to_2024-10-22_Azure.csv.gz' }
)

foreach ($item in $literature) {
    if ($Only -and $item.File -notlike "*$Only*") {
        continue
    }
    try {
        Download-Source -Url $item.Url -Destination (Join-Path $literatureDir $item.File)
    }
    catch {
        Write-Warning $_.Exception.Message
    }
}

foreach ($item in $datasets) {
    if ($Only -and $item.File -notlike "*$Only*") {
        continue
    }
    Download-Source -Url $item.Url -Destination (Join-Path $datasetDir $item.File)
}

Write-Host 'All requested downloads attempted.'
