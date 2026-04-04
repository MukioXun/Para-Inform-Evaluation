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

    # emotion2vec_plus_large 官方情感标签映射 (0-8)
    EMOTION_MAP = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "other",
        6: "sad",
        7: "surprised",
        8: "unknown"
    }

    # 中文情感映射
    EMOTION_CN_MAP = {
        "生气": "angry",
        "愤怒": "angry",
        "高兴": "happy",
        "开心": "happy",
        "快乐": "happy",
        "中性": "neutral",
        "平静": "neutral",
        "悲伤": "sad",
        "伤心": "sad",
        "难过": "sad",
        "恐惧": "fearful",
        "害怕": "fearful",
        "厌恶": "disgusted",
        "讨厌": "disgusted",
        "惊讶": "surprised",
        "吃惊": "surprised",
        "其他": "other",
        "未知": "unknown"
    }

    # 情感类别列表
    EMOTION_CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

    def load_model(self):
        """加载emotion2vec模型"""
        from funasr import AutoModel

        print(f"  [ER] Loading Emotion2Vec model...")

        # 尝试使用本地模型路径或下载
        model_path = self.config.get('model_path', 'iic/emotion2vec_plus_large')

        try:
            # 使用FunASR的emotion2vec模型
            self.model = AutoModel(
                model=model_path,
                device=self.device,
                disable_update=True,
                disable_log=True
            )
            print(f"  [ER] Emotion2Vec loaded from: {model_path}")
        except Exception as e:
            print(f"  [ER] Failed to load from {model_path}: {e}")
            print(f"  [ER] Trying to download from modelscope...")

            # 尝试直接下载
            try:
                self.model = AutoModel(
                    model="iic/emotion2vec_plus_large",
                    device=self.device,
                    disable_update=True,
                    disable_log=True,
                    hub='ms'  # 使用 modelscope
                )
                print(f"  [ER] Emotion2Vec loaded from modelscope")
            except Exception as e2:
                print(f"  [ER] Failed to load emotion model: {e2}")
                print(f"  [ER] Will use fallback mode (random emotion)")
                self.model = None

        self.emotion_classes = self.config.get('emotion_classes', self.EMOTION_CLASSES)

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行情感识别"""
        # 检查模型是否加载成功
        if self.model is None:
            return self._fallback_annotate(audio_path)

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
        emotion_id = 8  # unknown的id
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
                primary_emotion = self._parse_emotion_label(raw_label)

                # 过滤掉 <unk> 等特殊标签
                if primary_emotion in ['<unk>', 'unk']:
                    primary_emotion = "unknown"

                # 获取emotion_id
                emotion_id = self._get_emotion_id(primary_emotion)

                # 构建分布
                for i, label in enumerate(labels):
                    if i < len(scores):
                        emo = self._parse_emotion_label(label)
                        if emo not in ['<unk>', 'unk']:
                            emotion_distribution[emo] = float(scores[i])

        if not emotion_distribution:
            emotion_distribution = {"unknown": 0.5}

        # 维度情感映射
        valence, arousal = self._map_to_vad(primary_emotion)

        predictions = {
            "discrete": {
                "emotion_id": emotion_id,
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
            "emotion_id": emotion_id,
            "primary_emotion": primary_emotion,
            "confidence": float(confidence)
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _parse_emotion_label(self, raw_label: str) -> str:
        """解析情感标签，支持中英文格式"""
        if '/' in raw_label:
            # 格式: "生气/angry"
            cn_part, en_part = raw_label.split('/', 1)
            return en_part.lower().strip()
        else:
            label = raw_label.strip()
            # 检查是否是中文
            if label in self.EMOTION_CN_MAP:
                return self.EMOTION_CN_MAP[label]
            return label.lower()

    def _get_emotion_id(self, emotion: str) -> int:
        """根据情感名称获取ID"""
        for id, name in self.EMOTION_MAP.items():
            if name == emotion.lower():
                return id
        return 8  # unknown

    def _map_to_vad(self, emotion: str) -> tuple:
        """将离散情感映射到VAD维度 (valence, arousal)"""
        vad_mapping = {
            "happy": (0.8, 0.6),
            "sad": (-0.6, 0.2),
            "angry": (-0.5, 0.8),
            "fearful": (-0.6, 0.7),
            "neutral": (0.0, 0.3),
            "calm": (0.2, 0.1),
            "disgusted": (-0.4, 0.4),
            "surprised": (0.3, 0.7),
            "other": (0.0, 0.5),
            "unknown": (0.0, 0.5)
        }
        return vad_mapping.get(emotion.lower(), (0.0, 0.5))

    def _fallback_annotate(self, audio_path: str) -> Dict[str, Any]:
        """降级处理：模型不可用时返回默认值"""
        predictions = {
            "discrete": {
                "emotion_id": 8,
                "primary_emotion": "unknown",
                "confidence": 0.0,
                "emotion_distribution": {"unknown": 1.0}
            },
            "dimensional": {
                "valence": 0.0,
                "arousal": 0.5,
                "dominance": 0.5
            }
        }

        logits_dict = {
            "emotion_id": 8,
            "primary_emotion": "unknown",
            "confidence": 0.0,
            "error": "emotion model not available"
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }


# 别名
HuBERTEmotionAnnotator = Emotion2VecAnnotator
ERAnnotator = Emotion2VecAnnotator
