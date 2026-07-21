# 第0天环境状态

## 本机已验证的配置

- 操作系统终端：Windows PowerShell 5.1
- 编辑器：VS Code（已在本机路径中可用）
- 版本控制：Git 2.45.1；项目已初始化为本地`main`分支
- 项目解释器：`.venv\\Scripts\\python.exe`
- Python：3.12.13（以本机 Codex Python 运行时创建的虚拟环境）
- 原始数据：`Dataset/`内三份`BurstGPT_v2.0_without_fails` CSV 已确认存在

为复用本机已验证的 Pandas 3.0.1 与 NumPy 2.3.5，虚拟环境设置为可访问该 Python 运行时的系统包；项目专用的第三方包则安装在`.venv/`中。

## 已导入验证的项目依赖

| 类别 | 已验证包 |
|---|---|
| 数据与存储 | pandas 3.0.1、numpy 2.3.5、pyarrow 25.0.0、PyYAML 6.0.3 |
| 建模与统计 | scikit-learn 1.9.0、XGBoost 3.3.0、SciPy 1.18.0、statsmodels 0.14.6 |
| 图形与解释 | matplotlib 3.11.1、seaborn 0.13.2、SHAP 0.52.0 |
| Notebook 内核 | ipykernel 7.3.0（由 VS Code Jupyter 扩展使用） |

## TensorFlow 状态

`tensorflow==2.20.0`已写入`requirements.txt`，并且该版本官方支持 Python 3.12；但在本机当前网络环境下，pip 在解析/下载该包时长期停滞且没有写入 wheel，因此尚未安装。其余第0天任务及 XGBoost/EDA 环境已完成。

待网络正常时，在项目根目录执行下列命令即可补齐 LSTM 环境：

```powershell
.\\.venv\\Scripts\\Activate.ps1
python -m pip install tensorflow==2.20.0
```

若常规索引仍然卡住，可使用已核实的 PyPI 官方 Windows/Python 3.12 wheel：

```powershell
python -m pip install https://files.pythonhosted.org/packages/f9/37/b97abb360b551fbf5870a0ee07e39ff9c655e6e3e2f839bc88be81361842/tensorflow-2.20.0-cp312-cp312-win_amd64.whl
```

安装后应执行`python -c "import tensorflow as tf; print(tf.__version__)"`，再运行 LSTM 脚本。
