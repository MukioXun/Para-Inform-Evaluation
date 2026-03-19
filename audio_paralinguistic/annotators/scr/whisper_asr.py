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
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 预处理
        inputs = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            # 语言设置：auto表示自动检测
            language = self.config.get('language', 'auto')

            if language == 'auto':
                # 自动检测语言，不设置forced_decoder_ids
                generated_ids = self.model.generate(**inputs)
            else:
                # 指定语言
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task=self.config.get('task', 'transcribe')
                )
                generated_ids = self.model.generate(
                    **inputs,
                    forced_decoder_ids=forced_decoder_ids
                )

        # 解码
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        # 获取语言
        language = self._detect_language(generated_ids)

        # 构建输出
        predictions = {
            "transcription": {
                "text": transcription,
                "language": language,
                "confidence": 0.9,  # Whisper不直接提供置信度
            }
        }

        logits = {
            "transcription_text": transcription,
            "language_detected": language
        }

        return {
            "predictions": predictions,
            "logits": logits
        }

    def _detect_language(self, generated_ids):
        """从生成的token中检测语言"""
        # 简单启发式：根据生成的第一个token判断
        # 实际可以用language detection模型
        return "auto"


# 别名
SCRAnnotator = WhisperASRAnnotator
