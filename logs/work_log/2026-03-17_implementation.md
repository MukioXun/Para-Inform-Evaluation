# 工作日志 - 2026-03-17

## 模型实现与环境配置

### 1. 环境配置

创建了新的conda环境并安装了所有必要依赖：

```bash
# 创建环境
conda create -n audio_paraling python=3.10 -y

# 激活环境
conda activate audio_paraling

# 安装PyTorch (CUDA 11.8)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install transformers librosa soundfile scikit-learn hdbscan umap-learn seaborn matplotlib tqdm requests

# 安装FunASR (SenseVoiceSmall)
pip install funasr modelscope
```

**环境信息：**
- Python: 3.10
- PyTorch: 2.7.1+cu118
- CUDA: 11.8
- Transformers: 5.3.0
- FunASR: 1.3.1

### 2. 模型实现

#### 2.1 ASR标注器 (asr_annotator.py)

**模型：** SenseVoiceSmall (iic/SenseVoiceSmall)

**功能：**
- 语音转文字
- 多语言支持（中英粤日）
- 自动情感识别
- 音频事件检测

**输出示例：**
```python
{
    "text": "转写文本",
    "language": "zh",
    "emotion": "neutral",
    "event": "speech",
    "confidence": 0.95
}
```

#### 2.2 说话人识别标注器 (speaker_annotator.py)

**模型：** SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)

**功能：**
- 提取说话人embedding (192维)
- 说话人验证（相似度计算）
- 批量处理支持

**输出示例：**
```python
{
    "speaker_embedding": [...],  # 192维向量
    "embedding_dim": 192,
    "speaker_confidence": 1.0
}
```

#### 2.3 情感识别标注器 (emotion_annotator.py)

**模型：** Wav2Vec2 (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)

**功能：**
- 8类情感识别 (happy, sad, angry, neutral, fear, surprise, disgust, calm)
- 情感分布输出
- 非口语情感检测（笑声、叹气等）
- 情感强度计算

**输出示例：**
```python
{
    "emotion_category": "happy",
    "emotion_confidence": 0.85,
    "emotion_distribution": {"happy": 0.85, "neutral": 0.10, ...},
    "non_verbal_emotion": "laughter",
    "emotion_intensity": 0.7
}
```

#### 2.4 VAD情感维度标注器 (vad_annotator.py)

**方法：** 基于声学特征的规则方法

**功能：**
- Valence（效价）：负面(-1) → 正面(+1)
- Arousal（唤醒度）：平静(-1) → 激动(+1)
- Dominance（支配度）：被动(-1) → 主动(+1)
- 情感象限分类

**输出示例：**
```python
{
    "valence": 0.5,
    "arousal": 0.3,
    "dominance": 0.1,
    "vad_confidence": 0.85,
    "emotion_quadrant": "high_arousal_positive"
}
```

#### 2.5 副语言特征标注器 (paralingual_annotator.py)

**模型：** Wav2Vec2-base + Librosa声学特征

**功能：**
- 语速估计
- 音高特征（均值、标准差、范围）
- 能量特征
- 停顿比例
- 音质判断
- 声学环境识别

**输出示例：**
```python
{
    "speech_rate": 4.2,
    "pitch_mean": 150.5,
    "pitch_std": 30.2,
    "pause_ratio": 0.15,
    "voice_quality": "clear",
    "acoustic_environment": "indoor"
}
```

### 3. 主入口更新 (main.py)

**支持的命令：**

```bash
# 单音频标注
python main.py annotate --input audio.wav --output result.json

# 批量处理
python main.py batch --input_dir ./audio/ --output ./results/annotations.jsonl

# 聚类分析
python main.py cluster --input ./results/annotations.jsonl --output_dir ./output/

# 完整流程
python main.py pipeline --input_dir ./audio/ --output_dir ./output/
```

**可选参数：**
- `--device`: cuda/cpu
- `--annotators`: asr, emotion, speaker, vad, paralingual
- `--num_workers`: 并行线程数
- `--use_qwen`: 启用千帆大模型辅助分析

### 4. 文件更新列表

| 文件 | 更新内容 |
|------|----------|
| `annotators/semantic/asr_annotator.py` | 实现SenseVoiceSmall集成 |
| `annotators/acoustic/speaker_annotator.py` | 实现SpeechBrain说话人识别 |
| `annotators/acoustic/emotion_annotator.py` | 实现Wav2Vec2情感识别 |
| `annotators/acoustic/vad_annotator.py` | 实现VAD声学特征分析 |
| `annotators/acoustic/paralingual_annotator.py` | 实现副语言特征提取 |
| `main.py` | 集成所有标注器，完善CLI |
| `requirements.txt` | 更新依赖清单 |

### 5. 待安装依赖

以下依赖需要按需安装：

```bash
# SpeechBrain (说话人识别)
pip install speechbrain

# 如需使用Pyannote
pip install pyannote.audio
```

### 6. 下一步计划

1. 音频数据位置：/home/u2023112559/qix/datasets/PASM_Lite
2. 运行完整pipeline测试
3. 验证各模型输出格式
4. 根据聚类结果调整参数
5. 完善文档和使用说明

---

*工作完成时间: 2026-03-17*
