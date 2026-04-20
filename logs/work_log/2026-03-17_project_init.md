# 工作日志

## 2026-03-17 项目框架搭建

### 完成内容

#### 1. 目录结构创建
创建了完整的项目目录结构：
```
audio_paralinguistic/
├── config/                     # 配置模块
├── core/                       # 核心模块
├── annotators/                 # 标注器模块
│   ├── semantic/              # 语义维度标注器
│   └── acoustic/              # 声学维度标注器
├── fusion/                     # 特征融合模块
├── clustering/                 # 聚类分析模块
├── evaluation/                 # 评估模块
├── api/                        # API模块
├── utils/                      # 工具模块
├── prompt/                     # 提示词库
├── data/                       # 数据目录
│   ├── input/
│   ├── intermediate/
│   └── output/
└── work_log/                   # 工作日志
```

#### 2. 配置模块 (config/)
- `model_config.py`: 模型配置类，定义各模型的路径和参数
- `feature_config.py`: 特征配置类，定义特征维度和融合权重

#### 3. 核心模块 (core/)
- `audio_processor.py`: 音频预处理器，支持格式转换、切分、重采样
- `feature_extractor.py`: 特征提取器基类
- `pipeline.py`: 主流程控制器，支持并行处理

#### 4. 标注器模块 (annotators/)
- `base_annotator.py`: 标注器基类，定义统一接口

**语义维度标注器:**
- `asr_annotator.py`: ASR标注器 (SenseVoiceSmall)
- `intent_annotator.py`: 意图识别标注器 (Phi-4-mini)
- `slu_annotator.py`: 口语解析标注器 (SLU-BART-small)

**声学维度标注器:**
- `speaker_annotator.py`: 说话人识别标注器 (CAM++/ECAPA-TDNN)
- `emotion_annotator.py`: 情感识别标注器 (Emotion2Vec+)
- `paralingual_annotator.py`: 副语言特征标注器 (w2v-BERT 2.0)
- `vad_annotator.py`: 多维情感标注器 (MERaLiON-SER)

#### 5. 特征融合模块 (fusion/)
- `feature_merger.py`: 多模型特征合并器
- `normalization.py`: 特征标准化器

#### 6. 聚类分析模块 (clustering/)
- `cluster_engine.py`: 聚类引擎，支持 K-Means/DBSCAN/HDBSCAN/层次聚类
- `dimension_reduction.py`: 降维模块，支持 PCA/t-SNE/UMAP
- `cluster_evaluator.py`: 聚类效果评估器

#### 7. 评估模块 (evaluation/)
- `schema_inducer.py`: 标注体系推导器
- `metrics.py`: 评估指标计算器
- `report_generator.py`: 报告生成器

#### 8. API模块 (api/)
- `qwen_client.py`: 千帆大模型客户端
- `llm_interface.py`: 统一LLM接口（支持Qwen/OpenAI/Gemini）

#### 9. 工具模块 (utils/)
- `audio_utils.py`: 音频处理工具函数
- `json_utils.py`: JSON处理工具函数
- `visualization.py`: 可视化工具函数

#### 10. 提示词模块 (prompt/)
- `annotation_prompts.py`: 标注相关提示词
- `evaluation_prompts.py`: 评估相关提示词

#### 11. 其他文件
- `main.py`: 主入口，支持 annotate/batch/cluster/pipeline 四种模式
- `requirements.txt`: 依赖清单（仅列出，未安装）

### 待完成工作

1. **模型实现**: 各标注器的 `load_model()` 和 `annotate()` 方法需要实际实现
2. **测试数据**: 准备测试音频数据
3. **环境配置**: 安装必要的依赖包
4. **模型下载**: 下载所需的小模型

### 设计决策

1. **框架优先**: 先搭建完整框架，模型实现留待后续填充
2. **接口统一**: 所有标注器继承基类，便于扩展
3. **并行支持**: Pipeline 支持多线程并行处理
4. **LLM可选**: 千帆大模型作为可选增强，不影响核心功能

### 参考项目
- Omni-Captioner: 参考了其代码组织方式和Agent设计
- api_captoer: 参考了API调用方式

---

*下次工作请继续在此文件追加日志*
