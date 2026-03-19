## 声学表征体系构建任务 (精简版)

### 1. 数据结构重构 (Schema)

**目标：** 建立三层嵌套结构，删除 `task`, `status`, `timestamp` 等冗余。

- **Top Level:** `audio_id`, `file_path`, `content_metadata`, `acoustic_features`
- **Sub-Level (`acoustic_features`):**
  - **`low_level`**: 基础物理特征。
  - **`embeddings`**: 模型中层表示。
  - **`high_level`**: 任务标签 (ER/SED/SAR)。
- **SED 压缩：** 仅保留 `top_events` 和 `prob_summary`，剔除 521 维全量 logits。

### 2. 特征补齐 (Low-Level)

| **维度**     | **关键字段 (Output)**                                    | **推荐工具**            |
| ------------ | -------------------------------------------------------- | ----------------------- |
| **Spectral** | MFCCs, Mel-stats, Centroid, Bandwidth, Rolloff, Flatness | `librosa`               |
| **Prosody**  | F0 (stats), Jitter, Shimmer, HNR                         | `pyworld` / `opensmile` |
| **Energy**   | RMS (stats), Loudness, Dynamic Range                     | `librosa`               |
| **Temporal** | Duration, Speech_rate, Pause_ratio, Voiced_ratio         | `VAD` / `Praat`         |
| **Timbre**   | Formants (F1-F3), Spectral_envelope, Harmonicity         | `Praat` / `librosa`     |

### 3. 深度表征与模型更新 (Models)

- **Embeddings:** 提取 `wav2vec2`, `HuBERT`, `CLAP` (Utterance-level, Mean Pooling)。

- **ER (情感):** 替换为 `emotion2vec_plus_large`，映射 0-8 标签 (emotion_map = {

  0: "angry",

  1: "disgusted",

  2: "fearful",

  3: "happy",

  4: "neutral",

  5: "other",

  6: "sad",

  7: "surprised",

  8: "unknown"

  })。

- **SAR (属性):** 路径指向 `/home/u2023112559/qix/Models/Models/SenseVoiceSmall`，输出 Gender, Age, Embedding。

- **SED (事件):** 重点保留 Speech, Breathing, Child speech。

### 4. Pipeline 与交付标准

- **流程：** Audio → Low_level → Embeddings → High_level → JSON Assemble。
- **规格：** 统一时间轴对齐；大数组使用 `float16` 存储；严禁 `NaN` 或全零字段。
- **定义：** 构建一个去冗余、多层级的统一声学表征数据生产管线。

