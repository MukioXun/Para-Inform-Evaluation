# 语音副语言感知评估系统 - 设计规划文档

## 一、项目概述

### 1.1 项目目标
构建一个语音特征分析系统，通过"预标注 + 聚类分析"的方法，反向推导标注体系，实现对语音模型副语言感知理解能力的评估。

### 1.2 核心方法论
```
音频输入 → 多模型预标注 → 特征融合 → 聚类分析 → 标注体系推导 → 评估能力构建
```
---

## 二、系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Audio Paralinguistic Analyzer                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   数据输入    │───▶│  多模型标注   │───▶│  特征融合    │───▶│ 聚类分析   │ │
│  │   Module     │    │   Pipeline   │    │   Module     │    │  Module    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ 音频预处理    │    │ 语义维度标注  │    │ 特征向量构建  │    │ K-Means   │ │
│  │ 格式转换      │    │ 声学维度标注  │    │ 标准化处理    │    │ DBSCAN    │ │
│  │ 切分处理      │    │ 情感维度标注  │    │ 降维可视化    │    │ 层次聚类   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         标注体系推导 & 评估模块                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ 类别归纳    │  │ 标签体系    │  │ 评估指标    │  │ 报告生成    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构设计

```
Audio_Captior/
├── audio_paralinguistic/           # 主项目目录
│   ├── config/                     # 配置文件
│   │   ├── __init__.py
│   │   ├── model_config.py         # 模型配置（API、路径等）
│   │   └── feature_config.py       # 特征维度配置
│   │
│   ├── core/                       # 核心模块
│   │   ├── __init__.py
│   │   ├── audio_processor.py      # 音频预处理
│   │   ├── feature_extractor.py    # 特征提取器基类
│   │   └── pipeline.py             # 主流程控制
│   │
│   ├── annotators/                 # 标注器模块（各小模型封装）
│   │   ├── __init__.py
│   │   ├── base_annotator.py       # 标注器基类
│   │   │
│   │   ├── semantic/               # A. 语义维度
│   │   │   ├── __init__.py
│   │   │   ├── asr_annotator.py    # ASR: SenseVoiceSmall / Faster-Whisper
│   │   │   ├── intent_annotator.py # 意图识别: Phi-4-mini / Gemma-3n-E2B
│   │   │   └── slu_annotator.py    # 口语解析: SLU-BART-small
│   │   │
│   │   └── acoustic/               # B. 语音/人物维度
│   │       ├── __init__.py
│   │       ├── speaker_annotator.py    # 人物身份: CAM++ / ECAPA-TDNN
│   │       ├── emotion_annotator.py    # 情感识别: Emotion2Vec+
│   │       ├── paralingual_annotator.py # 副语言特征: w2v-BERT 2.0
│   │       └── vad_annotator.py        # 多维情感: MERaLiON-SER
│   │
│   ├── fusion/                     # 特征融合模块
│   │   ├── __init__.py
│   │   ├── feature_merger.py       # 多模型特征合并
│   │   └── normalization.py        # 特征标准化
│   │
│   ├── clustering/                 # 聚类分析模块
│   │   ├── __init__.py
│   │   ├── cluster_engine.py       # 聚类引擎
│   │   ├── dimension_reduction.py  # 降维可视化
│   │   └── cluster_evaluator.py    # 聚类效果评估
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── schema_inducer.py       # 标注体系推导
│   │   ├── metrics.py              # 评估指标计算
│   │   └── report_generator.py     # 报告生成
│   │
│   ├── api/                        # API接口（千帆大模型）
│   │   ├── __init__.py
│   │   ├── qwen_client.py          # 千帆大模型客户端
│   │   └── llm_interface.py        # 统一LLM接口
│   │
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── audio_utils.py          # 音频处理工具
│   │   ├── json_utils.py           # JSON处理
│   │   └── visualization.py        # 可视化工具
│   │
│   ├── data/                       # 数据目录
│   │   ├── input/                  # 输入音频
│   │   ├── intermediate/           # 中间结果
│   │   └── output/                 # 最终输出
│   │
│   ├── prompt/                     # 提示词库
│   │   ├── __init__.py
│   │   ├── annotation_prompts.py   # 标注提示词
│   │   └── evaluation_prompts.py   # 评估提示词
│   │
│   ├── main.py                     # 主入口
│   └── requirements.txt            # 依赖清单
│
├── api_captoer/                    # 已有参考实现
├── Omni-Captioner/                 # 参考项目
└── task_inf.md                     # 任务说明
```

---

## 三、模块详细设计

### 3.1 数据输入模块 (audio_processor.py)

**功能职责：**
- 音频格式统一转换（wav/mp3/flac/ogg → 标准格式）
- 音频切分（长音频切片，建议30秒以内）
- 重采样（统一采样率 16kHz）
- 音频质量检测

**输出：**
```python
{
    "audio_id": "unique_id",
    "file_path": "path/to/audio.wav",
    "duration": 15.2,
    "sample_rate": 16000,
    "segments": [
        {"start": 0.0, "end": 15.2, "path": "segment_001.wav"}
    ]
}
```

### 3.2 多模型标注模块 (annotators/)

#### 3.2.1 标注器基类设计

```python
class BaseAnnotator:
    """标注器基类"""
    def __init__(self, model_name: str, config: dict):
        self.model_name = model_name
        self.config = config

    def load_model(self):
        """加载模型"""
        raise NotImplementedError

    def annotate(self, audio_path: str) -> dict:
        """执行标注"""
        raise NotImplementedError

    def get_features(self) -> dict:
        """获取提取的特征"""
        raise NotImplementedError
```

#### 3.2.2 语义维度标注器

| 标注器 | 模型 | 输出特征 |
|--------|------|----------|
| ASRAnnotator | SenseVoiceSmall | 转写文本、情感标签、语种、音频事件 |
| IntentAnnotator | Phi-4-mini | 意图类别、置信度 |
| SLUAnnotator | SLU-BART-small | 语义解析结果、ASR鲁棒性评估 |

#### 3.2.3 声学维度标注器

| 标注器 | 模型 | 输出特征 |
|--------|------|----------|
| SpeakerAnnotator | CAM++ / ECAPA-TDNN | 说话人embedding、身份聚类 |
| EmotionAnnotator | Emotion2Vec+ | 情感类别、情感强度、非口语情感（叹气、笑声） |
| ParalingualAnnotator | w2v-BERT 2.0 | 副语言特征向量、声学环境描述 |
| VADAnnotator | MERaLiON-SER | 效价(Valence)、唤醒度(Arousal)、情感连续值 |

### 3.3 特征融合模块 (fusion/)

**功能：**
- 多模型输出特征的统一编码
- 特征向量的标准化（z-score / min-max）
- 特征权重可配置

**输出结构：**
```python
{
    "audio_id": "unique_id",
    "feature_vector": np.array([...]),  # 融合后的特征向量
    "feature_dict": {
        "semantic": {...},
        "acoustic": {...},
        "emotion": {...}
    },
    "metadata": {
        "annotators_used": [...],
        "confidence_scores": {...}
    }
}
```

### 3.4 聚类分析模块 (clustering/)

#### 3.4.1 聚类算法选择

| 算法 | 适用场景 | 参数 |
|------|----------|------|
| K-Means | 已知类别数量的快速聚类 | n_clusters |
| DBSCAN | 发现任意形状簇，自动确定簇数量 | eps, min_samples |
| Agglomerative | 层次聚类，适合构建类别层级 | linkage, n_clusters |
| HDBSCAN | 改进的DBSCAN，更稳定 | min_cluster_size |

#### 3.4.2 降维可视化

- **PCA**: 线性降维，快速可视化
- **t-SNE**: 保持局部结构，适合展示
- **UMAP**: 平衡全局和局部结构

#### 3.4.3 聚类评估指标

- 轮廓系数 (Silhouette Score)
- Davies-Bouldin Index
- Calinski-Harabasz Index

### 3.5 标注体系推导模块 (evaluation/schema_inducer.py)

**核心流程：**

```
聚类结果 → 统计各类别特征分布 → 特征重要性分析 → 类别命名/标签推导 → 标注体系输出
```

**输出标注体系：**
```python
{
    "schema_version": "1.0",
    "categories": [
        {
            "category_id": 1,
            "category_name": "高唤醒正向情感",
            "feature_characteristics": {
                "arousal": {"mean": 0.85, "std": 0.1},
                "valence": {"mean": 0.72, "std": 0.15},
                "dominant_emotions": ["excited", "happy"]
            },
            "sample_count": 150,
            "representative_samples": [...]
        },
        ...
    ],
    "feature_importance": {
        "arousal": 0.35,
        "valence": 0.28,
        "pitch_variation": 0.15,
        ...
    }
}
```

### 3.6 千帆大模型API接口 (api/)

**设计原则：**
- 统一接口封装，支持多种LLM后端
- 支持批量处理和并发请求
- 完善的错误处理和重试机制

```python
class QwenClient:
    """千帆大模型客户端"""

    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def analyze_annotation(self, annotation_data: dict, prompt: str) -> str:
        """使用大模型分析标注结果"""
        pass

    def induce_category_name(self, cluster_features: dict) -> str:
        """推导类别名称"""
        pass

    def generate_evaluation_report(self, results: dict) -> str:
        """生成评估报告"""
        pass
```

---

## 四、数据流设计

### 4.1 单条音频处理流程

```
┌─────────────┐
│  音频输入    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────────────────────────┐
│  音频预处理  │────▶│  格式转换、切分、重采样                │
└──────┬──────┘     └──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                      并行多模型标注                           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │    ASR    │ │  Intent   │ │  Speaker  │ │  Emotion  │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘   │
│        │             │             │             │          │
│        └──────────────┴──────────────┴─────────────┘        │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────┐
                    │    特征融合      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   保存中间结果    │
                    └─────────────────┘
```

### 4.2 批量处理与聚类流程

```
┌─────────────────────┐
│  所有音频标注结果     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    特征矩阵构建      │  [N_samples × M_features]
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    特征标准化        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    降维（可选）      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│              多种聚类算法并行             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ K-Means │ │ DBSCAN  │ │ HDBSCAN │   │
│  └────┬────┘ └────┬────┘ └────┬────┘   │
│       └───────────┴────────────┘        │
└───────────────────┬─────────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │   聚类效果评估    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   选择最优聚类    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   标注体系推导    │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   输出最终结果    │
          └─────────────────┘
```

---

## 五、接口设计

### 5.1 主入口 (main.py)

```python
"""
用法示例：
    # 单音频标注
    python main.py annotate --input audio.wav --output result.json

    # 批量处理
    python main.py batch --input_dir ./audio/ --output_dir ./results/

    # 聚类分析
    python main.py cluster --annotations ./results/annotations.jsonl --output ./output/schema.json

    # 完整流程
    python main.py pipeline --input_dir ./audio/ --output_dir ./output/
"""

# CLI 参数设计
parser.add_argument("--mode", choices=["annotate", "batch", "cluster", "pipeline"])
parser.add_argument("--input", help="输入音频文件或目录")
parser.add_argument("--output", help="输出文件或目录")
parser.add_argument("--annotators", nargs="+", help="指定使用的标注器")
parser.add_argument("--cluster_method", default="hdbscan", choices=["kmeans", "dbscan", "hdbscan", "agglomerative"])
parser.add_argument("--num_workers", type=int, default=4, help="并行处理数量")
parser.add_argument("--use_qwen", action="store_true", help="启用千帆大模型分析")
```

### 5.2 JSONL 输入格式

```json
{"id": "audio_001", "path": "/path/to/audio1.wav", "metadata": {...}}
{"id": "audio_002", "path": "/path/to/audio2.wav", "metadata": {...}}
```

### 5.3 JSONL 输出格式（标注结果）

```json
{
    "id": "audio_001",
    "annotations": {
        "asr": {"text": "...", "language": "zh", "emotion": "neutral"},
        "speaker": {"embedding": [...], "speaker_id": "S001"},
        "emotion": {"category": "happy", "confidence": 0.85, "arousal": 0.7, "valence": 0.8},
        "paralingual": {"features": [...], "environment": "indoor"}
    },
    "feature_vector": [...]
}
```

---

## 六、依赖清单 (requirements.txt)

```
# 音频处理
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0

# 深度学习框架
torch>=2.0.0
transformers>=4.35.0
funasr>=1.0.0          # SenseVoiceSmall

# ASR
faster-whisper>=0.9.0

# 说话人识别
wespeaker>=0.2.0       # CAM++ / ECAPA-TDNN

# 情感识别
emotion2vec>=0.1.0

# 聚类与机器学习
scikit-learn>=1.3.0
hdbscan>=0.8.0
umap-learn>=0.5.0

# 降维可视化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# API与并发
requests>=2.31.0
aiohttp>=3.8.0
tqdm>=4.65.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0

# 配置管理
pyyaml>=6.0
omegaconf>=2.3.0
```

---

## 七、开发阶段规划

### Phase 1: 基础框架搭建（预计3-5天）

| 任务 | 内容 |
|------|------|
| 1.1 | 项目目录结构创建 |
| 1.2 | 配置模块实现 |
| 1.3 | 音频预处理模块 |
| 1.4 | 标注器基类实现 |
| 1.5 | 主流程框架搭建 |

### Phase 2: 标注器实现（预计5-7天）

| 任务 | 内容 |
|------|------|
| 2.1 | ASR标注器（SenseVoiceSmall） |
| 2.2 | 情感识别标注器（Emotion2Vec+） |
| 2.3 | 说话人识别标注器（CAM++） |
| 2.4 | 多维情感标注器（MERaLiON-SER） |
| 2.5 | 意图识别标注器（可选） |

### Phase 3: 特征融合与聚类（预计3-4天）

| 任务 | 内容 |
|------|------|
| 3.1 | 特征融合模块 |
| 3.2 | 聚类引擎实现 |
| 3.3 | 降维可视化 |
| 3.4 | 聚类评估 |

### Phase 4: 标注体系推导（预计2-3天）

| 任务 | 内容 |
|------|------|
| 4.1 | 特征统计分析 |
| 4.2 | 千帆API集成 |
| 4.3 | 标注体系自动推导 |
| 4.4 | 报告生成 |

### Phase 5: 测试与优化（预计2-3天）

| 任务 | 内容 |
|------|------|
| 5.1 | 单元测试 |
| 5.2 | 集成测试 |
| 5.3 | 性能优化 |
| 5.4 | 文档完善 |

---

## 八、关键设计决策

### 8.1 模型选择策略

- **优先本地部署**: 对算力要求低的小模型优先本地部署，降低API成本
- **API备份**: 对于复杂分析任务，预留千帆API接口
- **模型版本**: 使用Distilled版本平衡效果和效率

### 8.2 特征设计原则

- **多层次**: 涵盖信号层、感知层、语义层、文化层
- **可解释**: 特征应具有明确的物理或心理意义
- **可融合**: 不同模型的输出应能统一编码

### 8.3 聚类策略

- **多算法对比**: 不依赖单一聚类算法
- **稳定性优先**: 选择稳定性高的聚类结果
- **可调整**: 支持用户干预聚类参数

---

## 九、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 模型依赖安装困难 | 开发延迟 | 使用Docker容器化部署 |
| 音频数据质量差 | 标注效果下降 | 增加数据预处理和过滤 |
| 聚类效果不理想 | 标注体系不可用 | 调整特征权重，尝试不同聚类算法 |
| API调用成本高 | 预算超支 | 批量处理，本地缓存结果 |

---

## 十、后续扩展方向

1. **实时处理**: 支持流式音频的实时标注
2. **多模态融合**: 结合视频信息进行综合分析
3. **主动学习**: 根据聚类结果选择样本进行人工标注
4. **模型微调**: 基于推导的标注体系微调专用模型

---

*文档版本: v1.0*
*创建时间: 2026-03-17*
*作者: Claude Code Assistant*
