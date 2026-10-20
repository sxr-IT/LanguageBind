# LanguageBind 复现说明

以语言为枢纽，把视频、音频、深度、红外映射到共享语义空间。

本仓库基于官方 [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind)，起始提交为 `7070c53375661cdb235801176b564b45f96f0648`。原论文发表于 ICLR 2024。原始代码和样例保留在仓库中，新增 `run_cpu_demo.py` 用于没有 NVIDIA CUDA 的电脑做小规模推理。

## 本机环境

- Windows 11，Intel Arc 140T，32 GB 内存。
- 独立 Python 3.10 环境：`D:\codex\envs\languagebind`。
- 安装官方指定的 PyTorch 1.13.1 系列 CPU 轮子，以及 `requirements.txt`。
- 模型权重默认缓存在 `D:\codex\languagebind_models`，不提交到 Git。

## 运行

在仓库根目录执行。首次运行需要下载约 3.4 GB 的图像和音频官方权重。若默认下载器长时间无进度，可先运行仓库内的并行下载脚本：

```powershell
D:\codex\envs\languagebind\Scripts\python.exe prepare_weights.py
```

再执行：

```powershell
D:\codex\envs\languagebind\Scripts\python.exe run_cpu_demo.py
```

这会载入仓库附带的两张图像和两段音频，计算它们与两条英文描述的余弦相似度，也会计算图像与音频的相似度。每行分数较高的列代表该样本更接近的文本。可选的视频分支运行方式：

```powershell
D:\codex\envs\languagebind\Scripts\python.exe run_cpu_demo.py --modalities image audio video
```

视频分支在 CPU 上可能明显更慢。原版 `inference.py` 固定使用 `cuda:0` 并一次载入五个非文本模态，适合有兼容 NVIDIA CUDA 环境时按官方说明运行。

## 解释结果

这个样例展示**共享语义空间中的检索与匹配**。相似度不等于风险概率，也不能单独证明某条安全规程适用。煤矿数据复现时，需要另行建立事件、场景和条款的人工核验标注。

## 安装复现

若在另一台机器安装，可先创建 Python 3.10 环境，再参考官方 `README.md` 安装 PyTorch 和 `requirements.txt`。本机无 NVIDIA CUDA，因此使用 CPU 轮子：

```powershell
uv venv D:\codex\envs\languagebind --python 3.10
uv pip install --python D:\codex\envs\languagebind\Scripts\python.exe "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" "torchaudio==0.13.1+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python D:\codex\envs\languagebind\Scripts\python.exe -r requirements.txt
uv pip install --python D:\codex\envs\languagebind\Scripts\python.exe "setuptools<81"
```

模型权重会在首次运行时从 Hugging Face 下载。建议先运行默认的图像和音频小样例，再尝试视频。
