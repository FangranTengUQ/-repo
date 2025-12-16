# Fangran 的实习项目指南 - AI 语音压缩

你好 Fangran！这份文档是我为你整理的项目说明，希望能帮你快速上手这个实习项目。

## 项目概览

这个项目的核心目标是**使用 AI 技术进行高效的语音/音频压缩**。

在传统的通信（如打电话、蓝牙音频）中，我们使用像 MP3, AAC, Opus 这样的算法。而你现在的任务是研究和使用 Meta (Facebook) 开发的 **Encodec** 模型。这是一种基于深度学习的“神经音频编解码器 (Neural Audio Codec)”。

**为什么这对“芯昇科技”很重要？**
芯昇科技（CMCC Chip）专注于芯片设计。在低功耗芯片（如物联网设备、智能穿戴）上，带宽非常宝贵。
*   **Encodec 的优势**：它能在极低的带宽下（比如 6kbps 甚至更低），提供比传统算法（MP3/Opus）好得多的音质。
*   **应用场景**：超低带宽的对讲机、高质量的蓝牙传输、卫星通信语音压缩。

---

## 文件夹结构说明

你的工作目录主要在 `4 Fangran/` 下：

```text
4 Fangran/
├── encodec/               # 核心代码库 (Meta Encodec)
│   ├── main.py            # [重要] 主运行脚本，你的入口点
│   ├── compress.py        # 压缩/解压缩的底层逻辑
│   ├── model.py           # 定义了神经网络模型的结构
│   ├── modules/           # 包含卷积层、LSTM等网络组件
│   ├── quantization/      # 量化模块 (核心技术，把连续信号变离散编码)
│   └── ...
├── arch.pdf               # 可能的模型架构说明文档
└── ...
```

---

## 核心代码解析 (`main.py`)

`main.py` 是你主要交互的文件。让我们一行行看它在做什么：

### 1. 初始化
```python
model = EncodecModel.encodec_model_24khz().to(device)
model.set_target_bandwidth(6.0)
```
*   这里加载了一个预训练好的模型，采样率是 24kHz（适合人声语音）。
*   `set_target_bandwidth(6.0)`：这是一个关键参数！
    *   它设置目标码率为 **6 kbps** (千比特每秒)。
    *   你可以尝试把它改成 `1.5` (1.5 kbps), `3.0` (3 kbps), 或 `12.0` (12 kbps) 来观察音质和压缩率的变化。
    *   **数值越小，压缩越狠，文件越小，但音质损失可能越大。**

### 2. 编码 (Encode) - 压缩过程
```python
encoded_frames = model.encode(wav)
```
*   **输入**：原始音频波形 (`wav`)，是一个巨大的数字矩阵。
*   **处理**：模型通过卷积神经网络提取特征，然后使用 **VQ (Vector Quantization，矢量量化)** 技术，将声音变成了“密码本”里的索引（一堆整数）。
*   **输出**：`encoded_frames`。这就是压缩后的数据，体积非常小，适合在无线信道上传输。

### 3. 解码 (Decode) - 还原过程
```python
reconstructed = model.decode(encoded_frames)
```
*   **输入**：压缩后的索引。
*   **处理**：模型根据索引查表，还原出特征，再通过神经网络生成回波形。
*   **输出**：`reconstructed`。这是还原后的音频，听起来应该和原声很像。

---

## 如何运行代码

目前环境已经配置好了。

1.  **准备音频**：
    把你想测试的 wav 音频文件放在 `4 Fangran/encodec/` 目录下，重命名为 `1.wav`（或者修改 `main.py` 里的 `input_file` 变量）。

2.  **运行**：
    在终端中输入：
    ```bash
    cd "4 Fangran/encodec"
    # 需要先设置 PYTHONPATH 以便 python 找到模块
    export PYTHONPATH=$PYTHONPATH:"/app/4 Fangran"
    python3 main.py
    ```

3.  **查看结果**：
    运行成功后，会生成 `1output.wav`。你可以下载下来听一听，对比一下和原文件的区别。

---

## 进阶思考 (给导师展示用)

如果你想给 mentor 留下好印象，可以尝试思考以下问题：

1.  **延时 (Latency)**：神经网络推理通常比较慢。在芯片上跑，延时是多少？（Encodec 有流式处理模式 Streaming，可以研究一下）。
2.  **复杂度**：这个模型有多大？参数量是多少？芯昇的芯片能跑得动吗？
3.  **抗噪性**：如果传输过程中（`encoded_frames`）丢了几个包，解调出来的声音会变成什么样？（这可能就是另一个文件夹 `llrNet` 存在的意义——研究信道传输和纠错）。

祝你实习顺利！如果有报错，记得看 `main.py` 里的 `try...except` 部分打印的错误信息。
