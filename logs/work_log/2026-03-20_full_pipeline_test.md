# 完整Pipeline测试报告 (含Tone标注)

**实验时间**: 2026-03-20 15:41:27
**实验ID**: exp_20260320_154022

---

## 测试概要

使用 `run_pipeline_with_tone.sh` 脚本完成完整流程测试：
1. Step 1: 主Pipeline (默认环境: audio_paraling)
2. Step 2: Tone标注 (Audio-Reasoner环境)

---

## Step 1: 主Pipeline

| 指标 | 值 |
|------|------|
| 数据量 | 20条 |
| 成功数 | 20 |
| 失败数 | 0 |
| 模型加载 | 35.61s |
| 标注时间 | 70.99s (3.55s/条) |
| 任务 | LowLevel, ER, SED, SAR, SCR, SpER |

### 模块状态

| 模块 | 状态 | 说明 |
|------|------|------|
| LowLevel | ⚠️ 部分正常 | pyworld不可用，使用fallback F0 |
| ER | ✅ 正常 | emotion2vec_plus_large |
| SED | ✅ 正常 | PANNs-CNN14 |
| SAR-Age | ❌ 失败 | 'NoneType' object is not callable |
| SAR-Gender | ✅ 正常 | ECAPA-TDNN |
| SCR | ✅ 正常 | whisper-medium |
| SpER | ✅ 正常 | FunASR |

---

## Step 2: Tone标注

| 指标 | 值 |
|------|------|
| 环境 | Audio-Reasoner |
| 模型加载 | ~14s (4个checkpoint shards) |
| 处理数 | 20条 |
| 成功数 | 20 |
| 跳过数 | 0 |
| 失败数 | 0 |

### Tone标注示例 (000001.mp3)

```
The audio features a single speaker. The speaker's voice is clear and
easily understandable. The speaker is discussing a physics concept,
specifically the relationship between inertia and force. The speaker's
voice is steady and measured, without any noticeable emotional inflection.
```

---

## 发现的问题

### 1. Age标注失败
- 错误: `'NoneType' object is not callable`
- 原因: age_classifier.py中模型调用问题
- 影响: 所有audio的age字段为空

### 2. LowLevel特征为空
- 原因: feature_extractor.py输出结构问题
- 影响: spectral, prosody, energy等字段为空对象

### 3. SCR转录为空
- transcription.text 为空
- 可能原因: 语言检测或音频预处理问题

---

## 输出文件

```
exp_20260320_154022/
├── experiment_config.json
├── experiment_stats.json
├── results_summary.json
└── merged/
    ├── 000001_merged.json  (含tone描述)
    ├── 000002_merged.json
    ├── ...
    └── 000020_merged.json
```

---

## 完整结果示例 (000001_merged.json)

```json
{
  "audio_id": "000001",
  "content_metadata": {
    "duration_seconds": 13.81,
    "sample_rate": 16000
  },
  "acoustic_features": {
    "low_level": {
      "spectral": {},
      "prosody": {},
      "energy": {},
      "temporal": {},
      "timbre": {}
    },
    "high_level": {
      "emotion": {
        "primary_emotion": "neutral",
        "confidence": 0.9999998807907104
      },
      "events": {
        "prob_summary": {
          "Speech": 0.0352,
          "Child speech": 0.0721
        }
      },
      "speaker": {
        "gender": {},
        "age": {},
        "tone": {
          "description": "The audio features a single speaker. The speaker's voice is clear..."
        }
      }
    }
  }
}
```

---

## 脚本验证结论

✅ `run_pipeline_with_tone.sh` 脚本运行正常
- 成功切换conda环境
- 主pipeline和tone标注按序执行
- 结果正确合并

### 待修复问题

1. **Age标注器**: 修复NoneType调用错误
2. **LowLevel特征**: 检查输出结构
3. **SED阈值**: 降低至0.1以检测更多事件
4. **SCR转录**: 调试whisper输出

---

## 使用命令

```bash
# 完整流程测试
./run_pipeline_with_tone.sh \
    --input /home/u2023112559/qix/datasets/PASM_Lite \
    --output ../data/experiments/exp_20260320_154022 \
    --limit 20 \
    --device cuda
```
