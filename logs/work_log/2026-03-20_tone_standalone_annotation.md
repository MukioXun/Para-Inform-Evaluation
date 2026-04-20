# Tone模块独立标注方案

**日期**: 2026-03-20
**任务**: 解决Tone模块环境依赖问题，实现独立标注

---

## 问题背景

Tone Annotator基于Audio-Reasoner (Qwen2-Audio)，需要swift模块：
- swift模块与主环境存在依赖冲突
- 直接在主pipeline中调用会导致环境问题

## 解决方案

采用**两阶段分离执行**策略：
1. 主pipeline在默认环境运行（排除tone）
2. Tone标注在Audio-Reasoner环境独立运行
3. 结果合并到统一的merged文件

---

## 实现文件

### 1. 独立Tone标注脚本

**文件**: `audio_paralinguistic/scripts/run_tone_annotation.py`

```bash
# 使用方法
conda activate Audio-Reasoner
python run_tone_annotation.py --input ./data/annotations/merged --audio-dirs /datasets/PASM_Lite
```

**功能**：
- `--mode merged`: 处理merged目录，补充缺失的tone标注
- `--mode audio`: 直接处理音频目录
- 自动检测哪些文件需要tone标注
- 更新merged文件中的tone字段

**核心逻辑**：
```python
def needs_tone_annotation(merged_data: Dict) -> bool:
    """检查是否需要tone标注"""
    speaker = merged_data.get('acoustic_features', {}).get('high_level', {}).get('speaker', {})
    tone = speaker.get('tone', {})
    if not tone:
        return True
    description = tone.get('description', '')
    return not description or description in ['unavailable', 'unknown', '']
```

### 2. 主启动脚本

**文件**: `audio_paralinguistic/scripts/run_pipeline_with_tone.sh`

```bash
# 完整流程
./run_pipeline_with_tone.sh --input /datasets/PASM_Lite --output ./data/annotations

# 仅运行tone标注
./run_pipeline_with_tone.sh --mode tone-only --input ./data/annotations/merged

# 跳过tone标注
./run_pipeline_with_tone.sh --input ./audio --output ./output --no-tone
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: single/batch/test/tone-only | batch |
| `--input` | 输入音频文件或目录 | 必填 |
| `--output` | 输出目录 | ./data/annotations |
| `--tasks` | 任务列表 | LowLevel,ER,SED,SAR,SCR,SpER |
| `--no-tone` | 跳过tone标注 | false |
| `--conda-env` | 主流程conda环境 | 当前环境 |
| `--tone-env` | tone标注环境 | Audio-Reasoner |

**执行流程**：
```
Step 1: 主Pipeline (默认环境)
    ├── LowLevel: 低级声学特征
    ├── ER: 情感识别
    ├── SED: 声学事件检测
    ├── SAR: 说话人属性 (Age + Gender, 排除Tone)
    ├── SCR: 语音内容
    └── SpER: 语音实体
    → 输出: merged/*.json (tone字段为空)

Step 2: Tone标注 (Audio-Reasoner环境)
    ├── 扫描merged目录
    ├── 检测缺失tone的文件
    ├── 执行Audio-Reasoner推理
    └── 更新merged文件
    → 输出: merged/*.json (tone字段已填充)
```

---

## 文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/run_tone_annotation.py` | 新建 | 独立Tone标注脚本 |
| `scripts/run_pipeline_with_tone.sh` | 新建 | 主启动脚本 |

---

## 使用示例

### 示例1: 批量处理数据集

```bash
# 完整流程
cd /home/u2023112559/qix/Project/Audio_Captior/audio_paralinguistic/scripts
./run_pipeline_with_tone.sh \
    --mode batch \
    --input /datasets/PASM_Lite \
    --output ./data/annotations \
    --device cuda
```

### 示例2: 仅补充Tone标注

```bash
# 已有merged结果，仅运行tone
./run_pipeline_with_tone.sh \
    --mode tone-only \
    --input ./data/annotations/merged \
    --tone-env Audio-Reasoner
```

### 示例3: 单文件测试

```bash
./run_pipeline_with_tone.sh \
    --mode single \
    --input /datasets/PASM_Lite/000001.mp3 \
    --output ./test_output
```

---

## 注意事项

1. **环境准备**：
   - 主环境: 需要PyTorch, transformers, funasr等
   - Audio-Reasoner环境: 需要swift模块

2. **音频目录**：
   - merged模式下需要指定`--audio-dirs`，用于根据audio_id查找音频文件

3. **限制处理数量**：
   - 使用`--limit N`限制处理数量，便于测试

---

## 后续优化

- [ ] 支持并行处理多个音频文件
- [ ] 添加进度条显示
- [ ] 支持断点续传
- [ ] 添加错误重试机制
