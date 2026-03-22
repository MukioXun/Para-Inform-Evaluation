# Pipeline优化实验总结

**日期**: 2026-03-20
**实验目标**: 根据TASK_add_model.md优化Pipeline，整合新的SAR子模块

---

## 1. 实验配置

### 1.1 模型配置

| 任务 | 模型 | 路径 | 状态 |
|------|------|------|------|
| LowLevel | librosa+pyworld | - | ✅ 正常 |
| ER | emotion2vec_plus_large | modelscope | ✅ 正常 |
| SED | PANNs-CNN14 | /Models/panns | ✅ 正常 |
| SAR-Age | wav2vec2-based | /Models/age-classification | ⚠️ 结果异常 |
| SAR-Gender | ECAPA-TDNN | /Models/gender-classifier | ✅ 正常 |
| SAR-Tone | Audio-Reasoner | /Models/Audio-Reasoner | ❌ 缺少swift |

### 1.2 测试数据

- 数据集: PASM_Lite
- 测试文件: 000001.mp3
- 时长: 13.81秒

---

## 2. 实验结果

### 2.1 模块加载成功率

| 模块 | 加载状态 | 备注 |
|------|---------|------|
| LowLevel | ✅ 成功 | VAD模型已加载 |
| ER | ✅ 成功 | emotion2vec加载成功 |
| SED | ✅ 成功 | PANNs weights loaded |
| SAR-Age | ✅ 成功 | Model loaded |
| SAR-Gender | ✅ 成功 | Model loaded |
| SAR-Tone | ❌ 失败 | No module named 'swift' |

### 2.2 推理结果分析

#### ER (情感识别)
```json
{
  "primary_emotion": "neutral",
  "confidence": 0.9999,
  "distribution": {
    "neutral": 0.9999998807907104,
    "happy": 5.2e-08,
    "sad": 5.4e-08
  }
}
```
**结论**: ER模块工作正常，高置信度识别为neutral。

#### SED (声学事件检测)
```json
{
  "top_events": [],
  "prob_summary": {
    "Speech": 0.0352,
    "Child speech": 0.0721
  },
  "primary_event": "unknown"
}
```
**问题**: 阈值0.5过高，导致无事件被检测。

#### SAR-Gender (性别识别)
```json
{
  "label": "male",
  "confidence": 0.9869
}
```
**结论**: Gender模块工作正常，高置信度识别为male。

#### SAR-Age (年龄预测)
```json
{
  "age_value": 0.62,
  "age_group": "child",
  "confidence": 0.6
}
```
**问题**: 预测年龄0.62岁不合理，可能是:
1. 模型输出需要缩放
2. 输入预处理有问题
3. 模型训练数据的年龄范围不同

---

## 3. 问题分析与解决方案

### 3.1 Tone模块环境依赖

**问题**: `No module named 'swift'`

**原因**: Audio-Reasoner使用swift框架进行推理

**解决方案**:
```bash
# 方案1: 切换到Audio-Reasoner环境
conda activate Audio-Reasoner

# 方案2: 安装swift
pip install swift
```

### 3.2 Age预测异常

**问题**: age_value=0.62岁

**可能原因**:
1. 模型输出是归一化值，需要反归一化
2. 参考原始代码，模型输出直接是年龄值

**解决方案**: 检查原始推理代码
```python
# 原始代码输出示例
# [[ 0.33793038 0.2715511  0.2275236  0.5009253 ]]
# 第一列是年龄
```

需要检查:
- 模型是否正确加载
- 输入预处理是否与训练时一致

### 3.3 SED阈值过高

**问题**: top_events为空

**解决方案**: 降低threshold或调整后处理逻辑

---

## 4. 输出结构验证

### 4.1 新结构符合预期

```json
{
  "audio_id": "000001",
  "acoustic_features": {
    "low_level": { ... },
    "high_level": {
      "emotion": { ... },
      "events": { ... },
      "speaker": {
        "gender": { ... },
        "age": { ... },
        "tone": { ... }
      }
    }
  }
}
```

**变化**:
- ✅ 删除了embeddings字段
- ✅ speaker包含gender, age, tone三个子字段
- ✅ 低级特征完整保留

### 4.2 Embeddings模块已移除

原结构中的embeddings字段已被删除，符合任务要求。

---

## 5. 总结

### 5.1 完成情况

| 任务项 | 状态 |
|--------|------|
| 删除Embeddings模块 | ✅ 完成 |
| 创建age_classifier | ✅ 完成 |
| 创建gender_classifier | ✅ 完成 |
| 创建tone_annotator | ✅ 完成 |
| 重构SAR标注器 | ✅ 完成 |
| 修改pipeline执行顺序 | ✅ 完成 |
| 更新model_config.py | ✅ 完成 |
| 测试新pipeline | ⚠️ 部分通过 |

### 5.2 待解决

1. **Tone模块**: 需要在Audio-Reasoner环境中运行
2. **Age预测**: 需要检查输出缩放问题
3. **SED阈值**: 需要调整以获得更好的检测效果

### 5.3 下一步

1. 使用 `conda activate Audio-Reasoner` 测试Tone模块
2. 修复Age预测的缩放问题
3. 批量测试验证整体效果
