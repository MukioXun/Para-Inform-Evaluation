# 音频评测系统使用说明

## 概述

本系统对音频文件进行多维度标注，包括：
- **ASR**: 语音转文字 (Whisper-medium)
- **EMO**: 情感识别 (emotion2vec)
- **AGE**: 年龄分类 (wav2vec2-based)
- **GND**: 性别分类 (ECAPA-TDNN)
- **TONE**: 语气分析 (Audio-Reasoner / Qwen2-Audio)

## 目录结构

```
audio/
├── age/           # 年龄相关音频
│   ├── 04-18-05-07_84_adult/
│   │   ├── user.wav          # 用户输入
│   │   ├── glm4.wav          # 模型输出
│   │   ├── gpt-4o-voice-mode.wav
│   │   └── ...
│   └── ...
├── emotion/       # 情感相关音频
├── gender/        # 性别相关音频
└── sarcasm/       # 讽刺相关音频
```

## 使用方法

### 方法一：完整流程（推荐）

```bash
cd /home/u2023112559/qix/Project/Final_Project/Audio_Captior/audio_paralinguistic/scripts
chmod +x run_eval.sh
./run_eval.sh
```

### 方法二：分步运行

#### Step 1: 基础标注 (ASR, EMO, AGE, GND)

```bash
conda activate audio_paraling

python run_evaluation.py \
    --input /path/to/audio \
    --output /path/to/output \
    --device cuda \
    --workers 4 \
    --skip-tone
```

#### Step 2: TONE 标注

```bash
conda activate Audio-Reasoner

python run_tone_annotation.py \
    --input /path/to/output \
    --audio /path/to/audio \
    --workers 4
```

### 测试模式

```bash
# 仅处理前 5 个目录
python run_evaluation.py --input ./audio --output ./output --limit 5 --skip-tone

# 仅处理前 10 个 TONE 任务
python run_tone_annotation.py --input ./output --audio ./audio --limit 10
```

## 输出格式

每个目录输出一个 JSON 文件：

```json
{
  "category": "age",
  "label": "adult",
  "dir_name": "04-18-05-07_84_adult",
  "pairs": [
    {
      "input": {
        "file": "user.wav",
        "model": "user",
        "annotation": {
          "ASR": "Can you tell me how to make my own beer?",
          "EMO": {"emotion": "neutral", "confidence": 0.85},
          "AGE": {"age_group": "senior", "confidence": 0.8},
          "GND": {"gender": "female", "confidence": 0.985},
          "TONE": {"description": "The speaker's tone is calm and inquisitive..."}
        }
      },
      "output": {
        "file": "glm4.wav",
        "model": "glm4",
        "annotation": {
          "ASR": "Making your own beer at home can be...",
          "EMO": {...},
          "AGE": {...},
          "GND": {...},
          "TONE": {...}
        }
      }
    }
  ]
}
```

## 环境要求

### audio_paraling 环境
- Python 3.10+
- PyTorch 2.0+
- transformers
- funasr
- librosa
- torchaudio

### Audio-Reasoner 环境
- Python 3.10+
- PyTorch 2.0+
- swift (ms-swift)
- transformers

## 脚本说明

| 脚本 | 功能 | 环境 |
|------|------|------|
| `run_evaluation.py` | 基础标注 (ASR/EMO/AGE/GND) | audio_paraling |
| `run_tone_annotation.py` | TONE 标注 (多线程) | Audio-Reasoner |
| `run_eval.sh` | 完整流程脚本 | 两者 |

## 已知问题

1. **EMO 模型下载失败**: emotion2vec 可能需要手动下载或配置网络
2. **TONE 标注较慢**: Audio-Reasoner 推理时间约 3-5 秒/音频

## 性能参考

- 基础标注 (48 目录): ~10 分钟 (4 workers)
- TONE 标注 (576 音频): ~30-45 分钟 (4 workers)

## 统计

| 类别 | 目录数 | 音频数 |
|------|--------|--------|
| age | 10 | 120 |
| emotion | 20 | 240 |
| gender | 10 | 120 |
| sarcasm | 8 | 96 |
| **总计** | **48** | **576** |
