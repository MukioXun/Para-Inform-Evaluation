# 语音副语言感知评估系统 - 设计规划文档

## 一、项目概述

### 1.1 项目目标
使用任务特化的小模型对无标签音频进行多维度标注，获取各任务维度下的结构化输出，为后续分析语音模型的副语言感知能力提供数据基础。

### 1.2 核心任务
```
音频输入 → 5个任务特化模型并行标注 → 结构化输出保存 → 输出分析 → 标注体系构建
```

### 1.3 标注任务列表

| 任务代号 | 任务名称 | 目标 |
|----------|----------|------|
| SCR | Speech Content Reasoning | 语音内容推理、意图理解 |
| SpER | Speech Entity Recognition | 语音实体识别（直接从语音识别命名实体） |
| SED | Sound Event Detection | 声学事件检测（环境音、非言语声） |
| ER | Emotion Recognition | 情感识别（离散/维度情感） |
| SAR | Speaker Attribute Recognition | 说话人属性识别（性别、年龄、口音等） |

---

## 二、系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Audio Multi-Task Annotation System                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │   音频输入    │                                                           │
│  │   (wav/mp3)  │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐│
│  │                        任务特化模型标注层 (Task-Specific Models)          ││
│  │                                                                           ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           ││
│  │  │       SCR       │  │      SpER       │  │       SED       │           ││
│  │  │ 语音内容推理     │  │ 语音实体识别     │  │ 声学事件检测     │           ││
│  │  │                 │  │                 │  │                 │           ││
│  │  │ Whisper-Medium  │  │ FunASR-NER      │  │ PANNs-CNN14     │           ││
│  │  │ + T5-Small      │  │                 │  │                 │           ││
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           ││
│  │           │                    │                    │                     ││
│  │  ┌────────┴────────┐  ┌────────┴────────┐                               ││
│  │  │        ER       │  │       SAR       │                               ││
│  │  │   情感识别       │  │  说话人属性识别   │                               ││
│  │  │                 │  │                 │                               ││
│  │  │ HuBERT-Emotion  │  │ ECAPA-TDNN      │                               ││
│  │  │                 │  │ + AttributeHead │                               ││
│  │  └────────┬────────┘  └────────┬────────┘                               ││
│  │           │                    │                                          ││
│  └───────────┴────────────────────┴──────────────────────────────────────────┘│
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                        结构化输出层 (Structured Output)                   │ │
│  │                                                                           │ │
│  │   各任务模型按统一格式输出：                                                │ │
│  │   - task_name: 任务标识                                                   │ │
│  │   - predictions: 标注结果                                                 │ │
│  │   - logits: 概率分布（保留用于后续分析）                                   │ │
│  │   - metadata: 模型信息、处理时间                                          │ │
│  │                                                                           │ │
│  └───────────────────────────────────┬───────────────────────────────────────┘ │
│                                      │                                        │
│                                      ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐│
│  │                           输出分析层                                       ││
│  │  1. 统计各任务输出分布                                                     ││
│  │  2. 分析输出结构特征                                                       ││
│  │  3. 确定后续聚类/融合策略                                                  ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构设计

```
Audio_Captior/
├── audio_paralinguistic/           # 主项目目录
│   ├── config/                     # 配置文件
│   │   ├── __init__.py
│   │   ├── model_config.py         # 模型配置（路径、设备等）
│   │   └── task_config.py          # 任务配置
│   │
│   ├── core/                       # 核心模块
│   │   ├── __init__.py
│   │   ├── audio_processor.py      # 音频预处理
│   │   ├── pipeline.py             # 主流程控制
│   │   └── output_manager.py       # 输出管理
│   │
│   ├── annotators/                 # 标注器模块
│   │   ├── __init__.py
│   │   ├── base_annotator.py       # 标注器基类
│   │   │
│   │   ├── scr/                    # Speech Content Reasoning
│   │   │   ├── __init__.py
│   │   │   ├── whisper_backend.py  # Whisper ASR后端
│   │   │   └── reasoning_head.py   # T5推理头
│   │   │
│   │   ├── sper/                   # Speech Entity Recognition
│   │   │   ├── __init__.py
│   │   │   └── funasr_ner.py       # FunASR-NER标注器
│   │   │
│   │   ├── sed/                    # Sound Event Detection
│   │   │   ├── __init__.py
│   │   │   └── panns_detector.py   # PANNs事件检测器
│   │   │
│   │   ├── er/                     # Emotion Recognition
│   │   │   ├── __init__.py
│   │   │   └── hubert_emotion.py   # HuBERT情感识别
│   │   │
│   │   └── sar/                    # Speaker Attribute Recognition
│   │       ├── __init__.py
│   │       └── ecapa_attribute.py  # ECAPA属性识别
│   │
│   ├── schemas/                    # 输出模式定义
│   │   ├── __init__.py
│   │   ├── scr_schema.py           # SCR输出模式
│   │   ├── sper_schema.py          # SpER输出模式
│   │   ├── sed_schema.py           # SED输出模式
│   │   ├── er_schema.py            # ER输出模式
│   │   └── sar_schema.py           # SAR输出模式
│   │
│   ├── analysis/                   # 输出分析模块
│   │   ├── __init__.py
│   │   ├── distribution_analyzer.py # 分布分析
│   │   ├── structure_analyzer.py    # 结构分析
│   │   └── correlation_analyzer.py  # 关联分析
│   │
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── audio_utils.py          # 音频处理工具
│   │   └── json_utils.py           # JSON处理
│   │
│   ├── data/                       # 数据目录
│   │   ├── input/                  # 输入音频
│   │   ├── annotations/            # 标注结果（按任务分目录）
│   │   │   ├── scr/
│   │   │   ├── sper/
│   │   │   ├── sed/
│   │   │   ├── er/
│   │   │   └── sar/
│   │   └── merged/                 # 合并后的结果
│   │
│   ├── main.py                     # 主入口
│   └── requirements.txt            # 依赖清单
│
├── model_proposal.md               # 模型提案文档
└── audio_paralinguistic_design.md  # 设计文档
```

---

## 三、任务模型详细设计

### 3.1 SCR: Speech Content Reasoning

#### 3.1.1 模型配置

```python
SCR_CONFIG = {
    'asr_backend': {
        'model': 'openai/whisper-medium',
        'device': 'cuda',
        'language': 'auto',  # 自动检测
    },
    'reasoning_head': {
        'model': 'google/flan-t5-base',
        'max_length': 512,
        'task_prompts': {
            'intent': "What is the speaker's intent?",
            'sentiment': "What is the sentiment of this speech?",
            'summary': "Summarize the main point:"
        }
    }
}
```

#### 3.1.2 输出结构

```python
{
    "audio_id": "audio_001",
    "task": "SCR",
    "predictions": {
        "transcription": {
            "text": "这个订单什么时候能送达",
            "language": "zh",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "这个订单什么时候能送达",
                    "tokens": ["这", "个", "订", "单", ...],
                    "confidence": 0.92
                }
            ]
        },
        "reasoning": {
            "intent": {
                "label": "inquiry",
                "confidence": 0.89,
                "description": "询问配送时间"
            },
            "sentiment": {
                "label": "neutral",
                "confidence": 0.75
            },
            "key_entities": ["订单", "送达"]
        }
    },
    "logits": {
        "intent_logits": [0.12, 0.05, 0.89, ...],  # 保留用于后续分析
        "sentiment_logits": [0.15, 0.75, 0.10]
    },
    "metadata": {
        "model_asr": "whisper-medium",
        "model_reasoning": "flan-t5-base",
        "inference_time": 1.23,
        "timestamp": "2026-03-18T10:30:00Z"
    }
}
```

### 3.2 SpER: Speech Entity Recognition

#### 3.2.1 模型配置

```python
SPER_CONFIG = {
    'model': 'damo/speech_timestamp_prediction',  # FunASR
    'device': 'cuda',
    'entity_types': ['PER', 'LOC', 'ORG', 'TIME', 'MONEY', 'PRODUCT'],
}
```

#### 3.2.2 输出结构

```python
{
    "audio_id": "audio_001",
    "task": "SpER",
    "predictions": {
        "entities": [
            {
                "entity_text": "北京",
                "entity_type": "LOC",
                "start_time": 0.5,
                "end_time": 0.8,
                "confidence": 0.94
            },
            {
                "entity_text": "明天下午",
                "entity_type": "TIME",
                "start_time": 1.2,
                "end_time": 1.8,
                "confidence": 0.88
            }
        ],
        "entity_summary": {
            "LOC": ["北京"],
            "TIME": ["明天下午"],
            "PER": [],
            "ORG": []
        }
    },
    "logits": {
        # 每个时间帧的实体概率分布
        "frame_logits": [[...], [...], ...]
    },
    "metadata": {
        "model": "FunASR-NER",
        "inference_time": 0.45
    }
}
```

### 3.3 SED: Sound Event Detection

#### 3.3.1 模型配置

```python
SED_CONFIG = {
    'model': 'PANNs-CNN14',
    'weights': 'qiuqiangkong/panns_inference',
    'device': 'cuda',
    'threshold': 0.5,
    'audio_length': 10.0,  # 秒
}
```

#### 3.3.2 输出结构

```python
{
    "audio_id": "audio_001",
    "task": "SED",
    "predictions": {
        "events": [
            {
                "event_name": "Speech",
                "onset": 0.0,
                "offset": 3.5,
                "confidence": 0.95
            },
            {
                "event_name": "Door",
                "onset": 1.2,
                "offset": 1.5,
                "confidence": 0.67
            }
        ],
        "event_distribution": {
            "Speech": 0.95,
            "Door": 0.67,
            "Music": 0.12,
            "Telephone": 0.05,
            "Laughter": 0.03
        },
        "environment_tag": "indoor_office"
    },
    "logits": {
        # 527类AudioSet标签的完整概率分布
        "class_probabilities": [0.001, 0.002, ..., 0.95, ...],
        "class_labels": ["Speech", "Child speech", ...]  # 对应的标签名
    },
    "metadata": {
        "model": "PANNs-CNN14",
        "num_classes": 527,
        "inference_time": 0.32
    }
}
```

### 3.4 ER: Emotion Recognition

#### 3.4.1 模型配置

```python
ER_CONFIG = {
    'model': 'superb/hubert-large-ls960-ft-er',
    'device': 'cuda',
    'output_format': 'both',  # 'discrete', 'dimensional', 'both'
    'emotion_classes': ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral', 'calm'],
}
```

#### 3.4.2 输出结构

```python
{
    "audio_id": "audio_001",
    "task": "ER",
    "predictions": {
        "discrete": {
            "primary_emotion": "neutral",
            "confidence": 0.82,
            "emotion_distribution": {
                "neutral": 0.82,
                "calm": 0.10,
                "happy": 0.03,
                "sad": 0.02,
                "angry": 0.01,
                "fearful": 0.01,
                "disgusted": 0.01,
                "surprised": 0.00
            }
        },
        "dimensional": {
            "valence": 0.12,      # 效价：轻微正向
            "arousal": 0.25,      # 唤醒度：较低
            "dominance": 0.45,    # 支配度：中等
        },
        "nonverbal_events": {
            "laughter": False,
            "sigh": False,
            "cough": False
        }
    },
    "logits": {
        "emotion_logits": [0.03, 0.02, 0.01, 0.01, 0.01, 0.00, 0.82, 0.10],
        "valence_raw": 0.12,
        "arousal_raw": 0.25
    },
    "metadata": {
        "model": "HuBERT-Large-Emotion",
        "inference_time": 0.28
    }
}
```

### 3.5 SAR: Speaker Attribute Recognition

#### 3.5.1 模型配置

```python
SAR_CONFIG = {
    'embedding_model': 'speechbrain/spkrec-ecapa-voxceleb',
    'attribute_heads': {
        'gender': {'classes': ['male', 'female', 'unknown']},
        'age_group': {'classes': ['child', 'young', 'middle', 'senior', 'unknown']},
        'accent': {'classes': ['native', 'non-native', 'unknown']}
    },
    'device': 'cuda',
}
```

#### 3.5.2 输出结构

```python
{
    "audio_id": "audio_001",
    "task": "SAR",
    "predictions": {
        "attributes": {
            "gender": {
                "label": "female",
                "confidence": 0.91
            },
            "age_group": {
                "label": "young",
                "confidence": 0.78
            },
            "accent": {
                "label": "native",
                "confidence": 0.65
            }
        },
        "speaker_embedding": {
            "vector": [0.12, -0.34, 0.56, ...],  # 192维embedding
            "dimension": 192
        }
    },
    "logits": {
        "gender_logits": [0.05, 0.91, 0.04],
        "age_logits": [0.05, 0.78, 0.12, 0.03, 0.02],
        "accent_logits": [0.65, 0.25, 0.10]
    },
    "metadata": {
        "model": "ECAPA-TDNN",
        "inference_time": 0.18
    }
}
```

---

## 四、统一输出格式

### 4.1 单任务输出文件格式

每个任务输出一个JSON文件，存储在对应目录下：

```
data/annotations/
├── scr/audio_001.json
├── sper/audio_001.json
├── sed/audio_001.json
├── er/audio_001.json
└── sar/audio_001.json
```

### 4.2 合并输出格式

所有任务结果合并为一条记录：

```python
{
    "audio_id": "audio_001",
    "file_path": "/path/to/audio.wav",
    "duration": 3.5,

    "annotations": {
        "SCR": { ... },  # SCR输出
        "SpER": { ... }, # SpER输出
        "SED": { ... },  # SED输出
        "ER": { ... },   # ER输出
        "SAR": { ... }   # SAR输出
    },

    "processing_info": {
        "total_time": 2.46,
        "tasks_completed": ["SCR", "SpER", "SED", "ER", "SAR"],
        "tasks_failed": [],
        "timestamp": "2026-03-18T10:30:00Z"
    }
}
```

---

## 五、标注器基类设计

### 5.1 BaseAnnotator

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

class BaseAnnotator(ABC):
    """标注器基类"""

    TASK_NAME: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = config.get('device', 'cuda')

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行标注"""
        pass

    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        完整处理流程，输出标准格式
        """
        start_time = time.time()

        # 执行标注
        predictions = self.annotate(audio_path)

        inference_time = time.time() - start_time

        # 构建标准输出
        output = {
            "audio_id": Path(audio_path).stem,
            "task": self.TASK_NAME,
            "predictions": predictions.get("predictions", {}),
            "logits": predictions.get("logits", {}),
            "metadata": {
                "model": self.config.get("model_name", "unknown"),
                "inference_time": round(inference_time, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }

        return output

    def save(self, output: Dict[str, Any], save_path: str):
        """保存结果"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
```

### 5.2 标注器实现示例

```python
# annotators/er/hubert_emotion.py

from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
import torch
import librosa

class HuBERTEmotionAnnotator(BaseAnnotator):
    """HuBERT情感识别标注器"""

    TASK_NAME = "ER"

    def load_model(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config['model']
        )
        self.model = HubertForSequenceClassification.from_pretrained(
            self.config['model']
        ).to(self.device)
        self.model.eval()

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)

        # 预处理
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # 解析结果
        emotion_classes = self.config.get('emotion_classes', [])
        pred_idx = probs.argmax()

        predictions = {
            "discrete": {
                "primary_emotion": emotion_classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "emotion_distribution": {
                    cls: float(prob)
                    for cls, prob in zip(emotion_classes, probs)
                }
            }
        }

        return {
            "predictions": predictions,
            "logits": {
                "emotion_logits": logits.tolist()
            }
        }
```

---

## 六、主流程控制

### 6.1 Pipeline设计

```python
# core/pipeline.py

from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

class AnnotationPipeline:
    """标注流水线"""

    def __init__(self, config: Dict):
        self.config = config
        self.annotators = {}

    def register_annotator(self, name: str, annotator: BaseAnnotator):
        """注册标注器"""
        self.annotators[name] = annotator

    def load_all_models(self):
        """加载所有模型"""
        for name, annotator in self.annotators.items():
            print(f"Loading {name} model...")
            annotator.load_model()

    def annotate_single(
        self,
        audio_path: str,
        tasks: Optional[List[str]] = None
    ) -> Dict:
        """单音频标注"""
        tasks = tasks or list(self.annotators.keys())
        results = {}

        for task in tasks:
            if task in self.annotators:
                results[task] = self.annotators[task].process(audio_path)

        return self._merge_results(audio_path, results)

    def annotate_batch(
        self,
        audio_dir: str,
        output_dir: str,
        tasks: Optional[List[str]] = None,
        num_workers: int = 4
    ):
        """批量标注"""
        audio_files = list(Path(audio_dir).glob("*.wav"))

        for audio_file in audio_files:
            result = self.annotate_single(str(audio_file), tasks)

            # 保存合并结果
            output_path = Path(output_dir) / f"{audio_file.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    def _merge_results(self, audio_path: str, results: Dict) -> Dict:
        """合并各任务结果"""
        return {
            "audio_id": Path(audio_path).stem,
            "file_path": audio_path,
            "annotations": results,
            "processing_info": {
                "tasks_completed": list(results.keys()),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }
```

### 6.2 CLI接口

```python
# main.py

import argparse
from config.model_config import MODEL_CONFIGS
from core.pipeline import AnnotationPipeline
from annotators.scr.whisper_backend import WhisperASRAnnotator
from annotators.sper.funasr_ner import FunASRNERAnnotator
from annotators.sed.panns_detector import PANNsDetector
from annotators.er.hubert_emotion import HuBERTEmotionAnnotator
from annotators.sar.ecapa_attribute import ECAPAAttributeAnnotator

def main():
    parser = argparse.ArgumentParser(description="Audio Multi-Task Annotation")
    parser.add_argument("--mode", choices=["single", "batch", "analyze"])
    parser.add_argument("--input", help="输入音频文件或目录")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--tasks", nargs="+",
                        choices=["SCR", "SpER", "SED", "ER", "SAR", "all"],
                        default=["all"])
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    # 初始化流水线
    pipeline = AnnotationPipeline({})

    # 注册标注器
    if args.tasks == ["all"] or "SCR" in args.tasks:
        pipeline.register_annotator("SCR", WhisperASRAnnotator(MODEL_CONFIGS["SCR"]))
    if args.tasks == ["all"] or "SpER" in args.tasks:
        pipeline.register_annotator("SpER", FunASRNERAnnotator(MODEL_CONFIGS["SpER"]))
    if args.tasks == ["all"] or "SED" in args.tasks:
        pipeline.register_annotator("SED", PANNsDetector(MODEL_CONFIGS["SED"]))
    if args.tasks == ["all"] or "ER" in args.tasks:
        pipeline.register_annotator("ER", HuBERTEmotionAnnotator(MODEL_CONFIGS["ER"]))
    if args.tasks == ["all"] or "SAR" in args.tasks:
        pipeline.register_annotator("SAR", ECAPAAttributeAnnotator(MODEL_CONFIGS["SAR"]))

    # 加载模型
    pipeline.load_all_models()

    # 执行标注
    if args.mode == "single":
        result = pipeline.annotate_single(args.input)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.mode == "batch":
        pipeline.annotate_batch(args.input, args.output, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
```

---

## 七、输出分析模块

### 7.1 分析流程

```
┌─────────────────────┐
│  加载所有标注结果    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  各任务输出统计分析  │
│  - 分布统计         │
│  - 置信度分布       │
│  - 缺失值统计       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  输出结构特征分析    │
│  - 字段完整性       │
│  - 数据类型一致性   │
│  - 嵌套层级分析     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  任务间关联分析      │
│  - SCR-ER关联      │
│  - SED-环境分析    │
│  - SAR-ER关联      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  输出分析报告        │
│  - 各任务统计摘要   │
│  - 数据质量评估     │
│  - 后续分析建议     │
└─────────────────────┘
```

### 7.2 分析报告输出

```python
{
    "analysis_report": {
        "total_samples": 1000,
        "analysis_timestamp": "2026-03-18T12:00:00Z",

        "task_statistics": {
            "SCR": {
                "samples_processed": 998,
                "samples_failed": 2,
                "avg_transcription_length": 15.3,
                "language_distribution": {"zh": 0.85, "en": 0.12, "other": 0.03},
                "intent_distribution": {"inquiry": 0.35, "statement": 0.40, ...},
                "avg_confidence": 0.89
            },
            "SED": {
                "samples_processed": 1000,
                "top_events": {"Speech": 0.95, "Music": 0.15, "Noise": 0.08},
                "avg_events_per_sample": 1.3
            },
            "ER": {
                "emotion_distribution": {"neutral": 0.45, "happy": 0.20, ...},
                "avg_arousal": 0.35,
                "avg_valence": 0.12
            },
            "SAR": {
                "gender_distribution": {"male": 0.48, "female": 0.50, "unknown": 0.02},
                "age_distribution": {"young": 0.45, "middle": 0.35, ...}
            }
        },

        "correlation_analysis": {
            "emotion_gender": {"female_happy": 0.25, "male_angry": 0.12},
            "intent_emotion": {"inquiry_neutral": 0.30, ...}
        },

        "data_quality": {
            "completeness": 0.98,
            "avg_confidence": 0.85,
            "recommendations": [
                "SCR任务有2个样本失败，建议检查音频质量",
                "ER任务的arousal值分布偏低，可能需要校准"
            ]
        },

        "next_steps": [
            "基于ER和SCR输出构建情感-意图联合特征",
            "对SED检测到的事件进行时序分析",
            "考虑使用聚类方法发现隐含模式"
        ]
    }
}
```

---

## 八、依赖清单

```txt
# 音频处理
librosa>=0.10.0
soundfile>=0.12.0
torchaudio>=2.0.0

# 深度学习框架
torch>=2.0.0
transformers>=4.35.0

# 任务模型
openai-whisper>=20230314    # SCR - ASR后端
funasr>=1.0.0               # SpER - FunASR NER

# SED - PANNs (需要单独安装)
# pip install panns-inference

# 情感识别
# HuBERT通过transformers加载

# 说话人识别
speechbrain>=0.5.0          # SAR - ECAPA-TDNN

# 数据处理
numpy>=1.24.0
pandas>=2.0.0
pydantic>=2.0.0

# 工具
tqdm>=4.65.0
requests>=2.31.0
```

---

## 九、开发阶段规划

### Phase 1: 基础框架 (1-2天)

| 任务 | 内容 |
|------|------|
| 1.1 | 项目目录结构创建 |
| 1.2 | 标注器基类实现 |
| 1.3 | 配置模块实现 |
| 1.4 | 输出管理模块 |

### Phase 2: 各任务标注器实现 (3-5天)

| 任务 | 内容 |
|------|------|
| 2.1 | SCR标注器 (Whisper + T5) |
| 2.2 | SpER标注器 (FunASR-NER) |
| 2.3 | SED标注器 (PANNs) |
| 2.4 | ER标注器 (HuBERT-Emotion) |
| 2.5 | SAR标注器 (ECAPA-TDNN) |

### Phase 3: 流水线与输出 (1-2天)

| 任务 | 内容 |
|------|------|
| 3.1 | Pipeline主流程实现 |
| 3.2 | 批量处理功能 |
| 3.3 | 输出格式验证 |

### Phase 4: 输出分析 (1-2天)

| 任务 | 内容 |
|------|------|
| 4.1 | 分布分析模块 |
| 4.2 | 关联分析模块 |
| 4.3 | 分析报告生成 |

### Phase 5: 测试与验证 (1天)

| 任务 | 内容 |
|------|------|
| 5.1 | 单元测试 |
| 5.2 | 集成测试 |
| 5.3 | 样本数据验证 |

---

## 十、模型来源汇总

| 任务 | 模型 | 来源 | 备注 |
|------|------|------|------|
| SCR-ASR | Whisper-Medium | OpenAI/HuggingFace | 转写+时间戳 |
| SCR-Reasoning | Flan-T5-Base | Google/HuggingFace | 意图/情感推理 |
| SpER | FunASR-NER | ModelScope/damo | 语音实体识别 |
| SED | PANNs-CNN14 | qiuqiangkong | 527类AudioSet |
| ER | HuBERT-Large-Emotion | SUPERB/HuggingFace | 情感识别 |
| SAR | ECAPA-TDNN | SpeechBrain | 说话人属性 |

---

*文档版本: v3.0*
*更新时间: 2026-03-18*
*变更说明: 基于5个任务特化模型重新设计架构*
