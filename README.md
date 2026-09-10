---
title: Video Translator With Voice Cloning And Subtitles
emoji: 🏢
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.38.1
python_version: 3.10.14
app_file: app.py
pinned: false
license: mit
---

# 🎥 Video Translator with Voice Cloning & Subtitles

视频自动翻译、音色克隆与双语字幕压制系统。结合 **OpenAI Whisper** 语音识别、**Google Translate** 文本翻译、**MeloTTS** 语音合成、**OpenVoice v2** 零样本音色克隆以及 **FFmpeg** 高清中日韩字幕压制。

---

## ⚡ Google Colab 快速运行指南

在 Google Colab 中新建 Notebook，依次运行以下两个单元格即可：

### 🔹 单元格 1：环境与依赖一键准备（仅需运行一次）

```bash
# 1. 克隆代码仓库并拉取 LFS 模型权重
!git clone https://github.com/aoxue908/Video-Translator-with-Voice-Cloning-and-Subtitles.git /content/Video-Translator-with-Voice-Cloning-and-Subtitles
%cd /content/Video-Translator-with-Voice-Cloning-and-Subtitles

# 2. 安装系统编译依赖、MeCab、中文字体库与 git-lfs
!apt-get update -qq && apt-get install -y -qq rustc cargo mecab libmecab-dev mecab-ipadic-utf8 fonts-noto-cjk fonts-wqy-zenhei git-lfs
!git lfs install
!git lfs pull

# 3. 安装 Python 全部依赖、MeloTTS 并下载 unidic 词库
!pip install -r requirements.txt
!pip install --no-deps git+https://github.com/myshell-ai/MeloTTS.git
!python -m unidic download
```

### 🔹 单元格 2：启动应用

```bash
%cd /content/Video-Translator-with-Voice-Cloning-and-Subtitles
!python app.py
```

> 💡 运行后日志中会输出 `Running on public URL: https://xxxx.gradio.live`，点击公网链接即可在浏览器中开始使用！

---

## 💻 本地部署指南 (Local Setup)

要在本地显卡（如 NVIDIA RTX 3060 Ti / 3090 / 4090 等）运行：

```bash
# 1. 克隆项目仓库
git clone https://github.com/aoxue908/Video-Translator-with-Voice-Cloning-and-Subtitles.git
cd Video-Translator-with-Voice-Cloning-and-Subtitles

# 2. 安装 Python 依赖
pip install -r requirements.txt
python -m unidic download

# 3. 启动本地服务
python app.py
```
启动后在浏览器中打开 `http://127.0.0.1:7860` 即可使用。

---

## ✨ 核心特性与修复

- 🎤 **高准确率语音识别**：内置 Whisper 模型与 Prompt 引导，支持普通话及多国语言精准识别与分词。
- 🗣️ **零样本音色克隆**：使用 OpenVoice v2 提取原视频说话人声音特征，还原原声音色。
- 🌏 **多语言翻译合成**：基于 MeloTTS 支持中、英、日、韩、法、西班牙等多国语言高清合成。
- 🔤 **中日韩字幕渲染**：集成 Linux CJK 中文字体包支持，解决 FFmpeg 字幕压制的 `□□□` 乱码问题。
- 🛠️ **全套 Python 3.13 / PyTorch 2.6 兼容**：内置新版 PyTorch 权重加载与 torchaudio 路径别名自动兼容补丁。
