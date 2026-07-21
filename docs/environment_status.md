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
| 建模与统计 | scikit-learn 1.9.0、XGBoost 3.3.0、TensorFlow/Keras 2.20.0、SciPy 1.18.0、statsmodels 0.14.6 |
| 图形与解释 | matplotlib 3.11.1、seaborn 0.13.2、SHAP 0.52.0 |
| Notebook 内核 | ipykernel 7.3.0（由 VS Code Jupyter 扩展使用） |

## TensorFlow 状态

`tensorflow==2.20.0`已安装到项目`.venv`并完成导入、模型训练及`.keras`重载验证。由于项目绝对路径较长，首次直接安装触发了 Windows 长路径限制；最终通过指向项目根目录的临时短路径 junction 完成安装，随后删除该 junction。项目解释器现在可直接运行 LSTM：

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__)"
.\.venv\Scripts\python.exe src/06_lstm.py --horizon 15
```
