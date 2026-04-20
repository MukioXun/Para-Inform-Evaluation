# Pipeline优化工作日志

**日期**: 2026-03-20
**任务**: 根据TASK_add_model.md优化Pipeline

---

## 完成的工作

### 1. 删除Embeddings模块

- 从model_config.py移除Embeddings配置
- 从pipeline.py移除Embeddings处理逻辑
- 从输出结构中移除embeddings字段

### 2. 创建新的SAR子模块

#### 2.1 Age Classifier (`annotators/sar/age_classifier.py`)
- 基于wav2vec2的年龄回归模型
- 使用 `/Models/age-classification` 模型
- 输出: age_value (数值) + age_group (年龄段分类)

#### 2.2 Gender Classifier (`annotators/sar/gender_classifier.py`)
- 基于ECAPA-TDNN的性别分类模型
- 使用 `/Models/gender-classifier` 模型
- 输出: male/female + confidence

#### 2.3 Tone Annotator (`annotators/sar/tone_annotator.py`)
- 基于Audio-Reasoner (Qwen2-Audio) 的语气识别
- 使用 `/Models/Audio-Reasoner` 模型
- 输出: 仅保留`<CAPTION>`标签内的语气描述
- **需要特殊环境**: `conda activate Audio-Reasoner`

### 3. 重构SAR标注器

- 创建新的`sar_annotator.py`整合Age/Gender/Tone三个子任务
- 支持单独调用各子任务
- 更新`__init__.py`导出

### 4. 更新配置文件

```python
"SAR": {
    "model_name": "SARAnnotator",
    "enable_age": True,
    "enable_gender": True,
    "enable_tone": True,
    "sub_configs": {
        "Age": {...},
        "Gender": {...},
        "Tone": {...}
    }
}
```

### 5. 修改Pipeline执行顺序

- 按标注维度依次加载模型
- 顺序: LowLevel -> HighLevel (ER, SED, SAR, SCR, SpER)

---

## 文件变更

| 文件 | 操作 |
|------|------|
| `annotators/sar/age_classifier.py` | 新建 |
| `annotators/sar/gender_classifier.py` | 新建 |
| `annotators/sar/tone_annotator.py` | 新建 |
| `annotators/sar/sar_annotator.py` | 新建 |
| `annotators/sar/__init__.py` | 更新 |
| `config/model_config.py` | 更新 |
| `core/pipeline.py` | 更新 |

---

## 测试结果

测试文件: `/datasets/PASM_Lite/000001.mp3`

**输出结构**:
```json
{
  "audio_id": "000001",
  "acoustic_features": {
    "low_level": { "spectral": {...}, "prosody": {...}, ... },
    "high_level": {
      "emotion": { "primary_emotion": "neutral", "confidence": 0.9999 },
      "events": { "top_events": [], "prob_summary": {...} },
      "speaker": {
        "gender": { "label": "male", "confidence": 0.9869 },
        "age": { "age_value": 0.62, "age_group": "child" },
        "tone": {}
      }
    }
  }
}
```

---

## 待解决问题

### 1. Tone模块环境依赖

Tone annotator需要swift模块，这是Audio-Reasoner的依赖。

**解决方案**:
- 使用 `conda activate Audio-Reasoner` 环境运行

### 2. Age预测结果异常

age_value=0.62岁，被分类为child。需要检查:
- 模型输入预处理是否正确
- 是否需要后处理缩放

### 3. SED无检测结果

top_events为空，可能阈值设置过高(0.5)

---

## 后续工作 (更新: 2026-03-20)

### 已完成: Tone模块独立标注脚本

为解决Tone模块需要特殊环境(Audio-Reasoner)的问题，创建了独立的标注脚本:

#### 1. 独立Tone标注脚本 (`scripts/run_tone_annotation.py`)

```bash
# 在Audio-Reasoner环境中运行
conda activate Audio-Reasoner
python run_tone_annotation.py --input ./data/annotations/merged --audio-dirs /datasets/PASM_Lite
```

功能:
- 扫描已完成的merged结果文件
- 为缺少tone标注的文件补充tone
- 自动更新合并结果
- 支持两种模式:
  - `--mode merged`: 处理merged目录
  - `--mode audio`: 直接处理音频文件

#### 2. 主启动脚本 (`scripts/run_pipeline_with_tone.sh`)

```bash
# 完整流程: 主pipeline + tone标注
./run_pipeline_with_tone.sh --input /datasets/PASM_Lite --output ./data/annotations

# 仅运行tone标注 (需要已有merged结果)
./run_pipeline_with_tone.sh --mode tone-only --input ./data/annotations/merged

# 跳过tone标注
./run_pipeline_with_tone.sh --input ./audio --output ./output --no-tone
```

工作流程:
1. 在默认环境运行主pipeline (排除tone)
2. 切换到Audio-Reasoner环境运行tone标注
3. 自动合并结果到merged文件

参数说明:
- `--mode`: single/batch/test/tone-only
- `--input`: 输入音频文件或目录
- `--output`: 输出目录
- `--tasks`: 任务列表 (默认: LowLevel,ER,SED,SAR,SCR,SpER)
- `--no-tone`: 跳过tone标注
- `--conda-env`: 主流程conda环境
- `--tone-env`: tone标注conda环境 (默认: Audio-Reasoner)

---

## 待完成

1. ~~在Audio-Reasoner环境中测试Tone模块~~ -> 已创建独立脚本
2. 调整Age模型的后处理逻辑
3. 调整SED的阈值参数
4. 批量测试100条数据验证效果
