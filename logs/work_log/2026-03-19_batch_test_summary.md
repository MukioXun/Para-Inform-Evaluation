# 工作日志 - 2026-03-19

## 上午: Pipeline完善与测试

### 完成的工作
1. 创建LowLevel特征提取器 (Spectral, Prosody, Energy, Temporal, Timbre)
2. 创建Embeddings提取器 (wav2vec2, HuBERT, CLAP)
3. 更新ER标注器emotion_map (0-8标签)
4. 更新SAR标注器使用SenseVoiceSmall
5. 更新SED输出压缩格式
6. 重构Pipeline三层嵌套结构

### 修复的问题
- torch/torchaudio版本匹配
- pyworld float64类型要求
- VAD分块处理
- SAR推理参数

## 下午: 批量测试

### 测试配置
- 数据集: PASM_Lite (2000条)
- 测试规模: 100条随机采样
- 任务: LowLevel, ER, SED, SAR
- 处理时间: 711.85秒 (~12分钟)

### 测试结果
| 模块 | 状态 | 备注 |
|------|------|------|
| LowLevel | ✅ 正常 | 所有特征提取成功 |
| ER | ✅ 正常 | 平均置信度0.962 |
| SED | ✅ 正常 | 语音检测率7% |
| SAR | ⚠️ 低识别率 | 93%返回unknown |

### 关键发现
1. Pipeline稳定性良好，100%成功率
2. 情感识别偏向neutral (69%)
3. 说话人属性识别率低
4. 平均处理时间约7秒/文件

## 输出文件

### 代码文件
- `annotators/lowlevel/feature_extractor.py`
- `annotators/embeddings/embedding_extractor.py`
- `annotators/sar/sensevoice_attribute.py`
- `scripts/batch_test.py`

### 数据文件
- `data/batch_test_results/batch_test_report.json`
- `data/batch_test_results/all_results.json`
- `data/batch_test_results/merged/*.json`

### 日志文件
- `work_log/2026-03-19_pipeline_v2.md`
- `exp_log/2026-03-19_batch100_test.md`

## 待跟进事项

1. **SAR模块调试**: 分析SenseVoiceSmall识别率低的原因
2. **SED阈值调整**: 考虑降低语音检测阈值
3. **ER分析**: 调研neutral高占比的数据集原因
4. **性能优化**: 考虑并行处理提速

## 环境信息

```
Python: 3.12
torch: 2.6.0+cu124
torchaudio: 2.6.0+cu124
CUDA: 可用
关键依赖: librosa, pyworld, funasr, modelscope, transformers
```
