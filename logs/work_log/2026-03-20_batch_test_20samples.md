# 批量测试实验报告

**实验时间**: 2026-03-20 15:22:12
**实验ID**: exp_20260320_152212

---

## 实验配置

| 参数 | 值 |
|------|------|
| 输入数据 | /home/u2023112559/qix/datasets/PASM_Lite |
| 输出目录 | data/experiments/exp_20260320_152212 |
| 数据量 | 20 条 |
| 设备 | cuda |
| 任务 | LowLevel, ER, SED, SAR, SCR, SpER |
| Tone标注 | 跳过 (需在Audio-Reasoner环境单独运行) |

---

## 性能统计

| 指标 | 值 |
|------|------|
| 模型加载时间 | 22.91s |
| 标注总时间 | 220.66s |
| 平均每条时间 | 11.03s |
| 成功数 | 20 |
| 失败数 | 0 |
| 成功率 | 100% |

---

## 各模块加载状态

| 模块 | 状态 | 说明 |
|------|------|------|
| LowLevel | ✅ 正常 | VAD模型加载成功 |
| ER | ✅ 正常 | emotion2vec_plus_large |
| SED | ✅ 正常 | PANNs-CNN14 (527类) |
| SAR | ✅ 正常 | Age + Gender (Tone跳过) |
| SCR | ✅ 正常 | whisper-medium |
| SpER | ✅ 正常 | FunASR |

---

## 样本输出分析 (000001.mp3)

### 基本信息
- 时长: 13.81秒
- 采样率: 16000Hz
- 格式: mp3

### 情感识别 (ER)
- 主情感: neutral
- 置信度: 99.99%
- 效价: 0.0
- 唤醒度: 0.3

### 声学事件检测 (SED)
- Speech概率: 3.52%
- Child speech概率: 7.21%
- 阈值过滤后无高置信度事件 (threshold=0.5)

### 说话人属性 (SAR)
| 属性 | 结果 | 置信度 |
|------|------|--------|
| 性别 | male | 98.69% |
| 年龄 | 62.18岁 (senior) | 80% |
| 语气 | 未标注 | - |

### 低级特征 (LowLevel)
- **基频(F0)**: 平均 133.77Hz, 标准差 24.76Hz
- **能量**: RMS平均 0.164, 峰值 -9.37dB
- **语速**: 约 0.44 音节/秒
- **语音比例**: 70.14%

---

## 发现的问题

### 1. SED阈值过高
- Speech概率仅3.52%，远低于0.5阈值
- 建议: 降低阈值至0.1或动态阈值

### 2. SCR转录为空
- transcription.text 为空
- 可能原因: 模型未正确识别语言或音频内容

### 3. Tone未标注
- 需要在Audio-Reasoner环境中单独运行

---

## 输出文件

```
exp_20260320_152212/
├── experiment_config.json    # 实验配置
├── experiment_stats.json     # 统计数据
├── results_summary.json      # 结果汇总
└── merged/                   # 合并结果
    ├── 000001_merged.json
    ├── 000002_merged.json
    ├── ...
    └── 000020_merged.json
```

---

## 后续工作

1. 运行Tone标注 (在Audio-Reasoner环境)
   ```bash
   conda activate Audio-Reasoner
   python scripts/run_tone_annotation.py --input data/experiments/exp_20260320_152212/merged --audio-dirs /home/u2023112559/qix/datasets/PASM_Lite
   ```

2. 调整SED阈值参数

3. 检查SCR转录问题

4. 扩大测试规模 (100条/全量)
