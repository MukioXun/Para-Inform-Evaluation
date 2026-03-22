"""
SCR标注器 - Whisper ASR
语音内容识别，仅ASR转写，禁用Reasoning
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any
from pathlib import Path

# 设置HuggingFace镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from ..base_annotator import BaseAnnotator


class WhisperASRAnnotator(BaseAnnotator):
    """Whisper ASR标注器"""

    TASK_NAME = "SCR"

    def load_model(self):
        """加载Whisper模型"""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        model_path = self.config.get('model_path')

        print(f"  [SCR] Loading Whisper from: {model_path}")

        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"  [SCR] Whisper loaded successfully")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行ASR转写"""

        # 1️⃣ 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 2️⃣ 预处理（✅ 加 attention_mask）
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True
        )

        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # 3️⃣ 推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language="zh",
                    task="transcribe"
                )
            # language = self.config.get('language', 'auto')
            # task = self.config.get('task', 'transcribe')

            # if language == 'auto':
            #     # ✅ 自动语言检测（新标准写法）
            #     generated_ids = self.model.generate(
            #         input_features,
            #         attention_mask=attention_mask
            #     )
            # else:
            #     # ✅ 新版写法（替代 forced_decoder_ids）
            #     generated_ids = self.model.generate(
            #         input_features,
            #         attention_mask=attention_mask,
            #         language=language,
            #         task=task
            #     )

        # 4️⃣ 解码
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # 5️⃣ 语言（你这里其实没真正实现）
        language_out = language if language != "auto" else "unknown"

        return {
            "predictions": {
                "transcription": {
                    "text": transcription,
                    "language": language_out,
                    "confidence": 0.9
                }
            },
            "logits": {
                "transcription_text": transcription,
                "language_detected": language_out
            }
        }


# 别名
SCRAnnotator = WhisperASRAnnotator
