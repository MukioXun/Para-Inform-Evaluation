# 环境分离配置报告

**日期**: 2026-03-20
**任务**: 检查并分离标注器运行环境

---

## 环境分离策略

根据依赖冲突，将标注器分为两个环境运行：

| 环境 | 运行模块 | 原因 |
|------|----------|------|
| **audio_paraling** | LowLevel, ER, SED, Gender, SCR, SpER | 标准PyTorch生态 |
| **Audio-Reasoner** | Age, Tone | 需要swift模块 + transformers兼容性 |

---

## 模块依赖分析

### audio_paraling环境

| 模块 | 依赖 | 状态 |
|------|------|------|
| LowLevel | librosa, pyworld (可选), silero-vad | ✅ 兼容 |
| ER | funasr, emotion2vec | ✅ 兼容 |
| SED | torch, torchaudio, PANNs | ✅ 兼容 |
| Gender | torch, torchaudio, ECAPA-TDNN | ✅ 兼容 |
| SCR | transformers, whisper | ✅ 兼容 |
| SpER | funasr | ✅ 兼容 |

### Audio-Reasoner环境

| 模块 | 依赖 | 原因 |
|------|------|------|
| Age | transformers, wav2vec2 | 模型加载需要特定transformers版本 |
| Tone | swift, qwen2-audio | 需要swift模块 |

---

## 文件变更

### 1. 新增: `scripts/run_sar_audio_reasoner.py`

独立的Age+Tone标注脚本，在Audio-Reasoner环境中运行：

```bash
conda activate Audio-Reasoner
python run_sar_audio_reasoner.py \
    --input ./data/annotations/merged \
    --audio-dirs /datasets/PASM_Lite

# 仅标注Age
python run_sar_audio_reasoner.py --input ./merged --audio-dirs ./audio --only age

# 仅标注Tone
python run_sar_audio_reasoner.py --input ./merged --audio-dirs ./audio --only tone
```

### 2. 更新: `scripts/run_test_batch.py`

新增参数：
- `--skip-age`: 跳过Age标注
- `--skip-tone`: 跳过Tone标注

### 3. 更新: `scripts/run_pipeline_with_tone.sh`

**两阶段执行流程**：

```
Step 1: 主Pipeline (audio_paraling环境)
├── LowLevel: 低级声学特征
├── ER: 情感识别
├── SED: 声学事件检测
├── SAR-Gender: 性别识别
├── SCR: 语音内容
└── SpER: 语音实体
→ 输出: merged/*.json (age和tone为空)

Step 2: Age+Tone标注 (Audio-Reasoner环境)
├── Age: 年龄预测
└── Tone: 语气描述
→ 输出: merged/*.json (age和tone已填充)
```

---

## 使用方法

### 完整流程

```bash
./run_pipeline_with_tone.sh \
    --input /home/u2023112559/qix/datasets/PASM_Lite \
    --output ./data/experiments/exp_20260320 \
    --limit 20
```

### 仅运行主Pipeline

```bash
./run_pipeline_with_tone.sh \
    --input ./audio \
    --output ./output \
    --no-sar-ar
```

### 仅运行Age+Tone标注

```bash
./run_pipeline_with_tone.sh \
    --input ./data/annotations/merged \
    --mode sar-ar-only
```

---

## 配置文件更新

### model_config.py (SAR部分)

```python
"SAR": {
    "model_name": "SARAnnotator",
    "device": DEFAULT_DEVICE,
    "sample_rate": 16000,
    "enable_age": True,    # 在Audio-Reasoner环境运行
    "enable_gender": True, # 在audio_paraling环境运行
    "enable_tone": True,   # 在Audio-Reasoner环境运行
    "sub_configs": {
        "Age": {...},
        "Gender": {...},
        "Tone": {...}
    }
}
```

---

## 注意事项

1. **环境顺序**: 必须先运行主Pipeline，再运行Age+Tone标注
2. **结果合并**: Age和Tone结果会自动更新到merged文件中
3. **错误处理**: 如果Age或Tone标注失败，不会影响其他模块的结果

---

## 测试验证

```bash
# 运行测试
cd /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/scripts
./run_pipeline_with_tone.sh \
    --input /home/u2023112559/qix/datasets/PASM_Lite \
    --output ../data/experiments/exp_test \
    --limit 5
```
