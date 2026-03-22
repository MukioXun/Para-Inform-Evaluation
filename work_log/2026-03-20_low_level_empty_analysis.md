# Low_level为空问题分析

**日期**: 2026-03-20

---

## 问题现象

| 实验 | low_level | gender | age | tone |
|------|-----------|--------|-----|------|
| exp_20260320_152212 | ✅ 有数据 | ✅ male | ✅ 62岁 | 空 |
| exp_20260320_154022 | ❌ 空对象 | ❌ 空 | ❌ 空 | ✅ 有值 |

---

## 根本原因

**参数不匹配问题**：

1. `run_pipeline_with_tone.sh`传入了`--skip-age`参数
2. 但当时的`run_test_batch.py`代码还没有这个参数
3. 参数未被识别，可能导致默认行为异常

**证据**：
- exp_20260320_154022的`experiment_config.json`中没有`skip_age`字段
- 只有`skip_tone: true`

---

## 代码位置分析

### 1. feature_extractor.py (正确)

```python
def annotate(self, audio_path: str) -> Dict[str, Any]:
    # 正常提取所有特征
    features = {}
    features["spectral"] = self._extract_spectral(audio, sr)
    features["prosody"] = self._extract_prosody(audio, sr)
    # ...
    return {"predictions": features, "logits": {}}
```

只有音频无效时才返回空特征，但测试音频是有效的。

### 2. pipeline.py (正确)

```python
def _build_nested_structure(...):
    if "LowLevel" in results and "predictions" in results["LowLevel"]:
        low_level_data = results["LowLevel"]["predictions"]
        structure["acoustic_features"]["low_level"] = {
            "spectral": low_level_data.get("spectral", {}),
            # ...
        }
```

正确地从LowLevel结果提取数据。

### 3. run_sar_audio_reasoner.py (正确)

```python
def update_merged(merged_path, age_result, tone_result):
    with open(merged_path, 'r') as f:
        data = json.load(f)  # 读取现有数据
    # 只修改speaker.age和speaker.tone
    # 不影响low_level
```

正确地保留现有数据，只更新特定字段。

---

## 修复方案

### 已完成
1. `run_test_batch.py`添加了`--skip-age`参数
2. 正确保存`skip_age`到配置文件

### 需要验证
重新运行完整测试：

```bash
./run_pipeline_with_tone.sh \
    --input /home/u2023112559/qix/datasets/PASM_Lite \
    --output ./data/experiments/exp_new_test \
    --limit 5
```

---

## 预期正确结果

```json
{
  "acoustic_features": {
    "low_level": {
      "spectral": { "mfcc": {...}, "mel": {...} },
      "prosody": { "f0": {...} },
      "energy": {...},
      "temporal": {...},
      "timbre": {...}
    },
    "high_level": {
      "speaker": {
        "gender": {"label": "male", "confidence": 0.98},
        "age": {"age_value": 62, "age_group": "senior"},
        "tone": {"description": "..."}
      }
    }
  }
}
```
