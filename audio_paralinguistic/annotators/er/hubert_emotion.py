"""
ER标注器 - Emotion2Vec (FunASR)
情感识别，使用FunASR的emotion2vec模型
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..base_annotator import BaseAnnotator


class Emotion2VecAnnotator(BaseAnnotator):
    """Emotion2Vec情感识别标注器"""

    TASK_NAME = "ER"

    # 情感类别 (emotion2vec 输出格式: 中文/英文)
    EMOTION_CLASSES = ["angry", "happy", "neutral", "sad", "unknown"]

    def load_model(self):
        """加载emotion2vec模型"""
        from funasr import AutoModel

        print(f"  [ER] Loading Emotion2Vec model...")

        # 使用FunASR的emotion2vec模型
        self.model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device=self.device,
            disable_update=True,
            disable_log=True
        )

        self.emotion_classes = self.config.get('emotion_classes', self.EMOTION_CLASSES)
        print(f"  [ER] Emotion2Vec loaded")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行情感识别"""
        # 加载音频
        wav, sr = librosa.load(audio_path, sr=self.sample_rate)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0).float()

        # FunASR推理
        result = self.model.generate(
            input=wav_tensor,
            output_dir=None
        )

        # 解析结果
        primary_emotion = "unknown"
        confidence = 0.5
        emotion_distribution = {}

        if result and len(result) > 0:
            res = result[0]
            # emotion2vec输出格式: labels, scores
            labels = res.get('labels', [])
            scores = res.get('scores', [])

            if labels and scores:
                # 找到分数最高的标签
                max_idx = np.argmax(scores)
                raw_label = labels[max_idx]
                confidence = float(scores[max_idx])

                # 解析标签格式: "生气/angry" -> "angry"
                if '/' in raw_label:
                    primary_emotion = raw_label.split('/')[1].lower()
                else:
                    primary_emotion = raw_label.lower()

                # 过滤掉 <unk> 等特殊标签
                if primary_emotion in ['<unk>', 'unk']:
                    primary_emotion = "unknown"

                # 构建分布
                for i, label in enumerate(labels):
                    if i < len(scores):
                        # 解析标签
                        if '/' in label:
                            emo = label.split('/')[1].lower()
                        else:
                            emo = label.lower()
                        if emo not in ['<unk>', 'unk']:
                            emotion_distribution[emo] = float(scores[i])

        if not emotion_distribution:
            emotion_distribution = {"unknown": 0.5}

        # 维度情感映射
        valence, arousal = self._map_to_vad(primary_emotion)

        predictions = {
            "discrete": {
                "primary_emotion": primary_emotion,
                "confidence": float(confidence),
                "emotion_distribution": emotion_distribution
            },
            "dimensional": {
                "valence": valence,
                "arousal": arousal,
                "dominance": 0.5
            }
        }

        logits_dict = {
            "primary_emotion": primary_emotion,
            "confidence": float(confidence)
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _map_to_vad(self, emotion: str) -> tuple:
        """将离散情感映射到VAD维度"""
        vad_mapping = {
            "happy": (0.8, 0.6),
            "sad": (-0.6, 0.2),
            "angry": (-0.5, 0.8),
            "fearful": (-0.6, 0.7),
            "neutral": (0.0, 0.3),
            "calm": (0.2, 0.1),
            "disgusted": (-0.4, 0.4),
            "surprised": (0.3, 0.7),
            "unknown": (0.0, 0.5)
        }
        return vad_mapping.get(emotion.lower(), (0.0, 0.5))


# 别名
HuBERTEmotionAnnotator = Emotion2VecAnnotator
ERAnnotator = Emotion2VecAnnotator
