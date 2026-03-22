"""
SAR标注器 - SenseVoiceSmall
说话人属性识别，提取Gender, Age, Embedding
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..base_annotator import BaseAnnotator


class SenseVoiceAttributeAnnotator(BaseAnnotator):
    """SenseVoiceSmall说话人属性标注器"""

    TASK_NAME = "SAR"

    def load_model(self):
        """加载SenseVoiceSmall模型"""
        from funasr import AutoModel

        model_path = self.config.get('model_path', '/home/u2023112559/qix/Models/Models/SenseVoiceSmall')

        print(f"  [SAR] Loading SenseVoiceSmall from: {model_path}")

        try:
            # 使用FunASR加载SenseVoice
            self.model = AutoModel(
                model=model_path,
                device=self.device,
                disable_update=True,
                disable_log=True,
                trust_repo=True
            )
            print(f"  [SAR] SenseVoiceSmall loaded")
        except Exception as e:
            print(f"  [SAR] Warning: Failed to load SenseVoice: {e}")
            print(f"  [SAR] Falling back to basic speaker encoder...")
            self._init_fallback_encoder()

    def _init_fallback_encoder(self):
        """初始化备用编码器"""
        import torch.nn as nn
        import torch.nn.functional as F

        class SimpleEncoder(nn.Module):
            def __init__(self, input_dim=80, hidden_dim=512, embedding_dim=192):
                super().__init__()
                self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.fc = nn.Linear(hidden_dim, embedding_dim)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = torch.mean(x, dim=2)
                return self.fc(x)

        self.model = SimpleEncoder()
        self.model.to(self.device)
        self.model.eval()
        self.use_fallback = True
        print(f"  [SAR] Fallback encoder initialized (embedding_dim=192)")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行说话人属性识别"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._annotate_fallback(audio)

        try:
            # 使用SenseVoice推理
            return self._annotate_sensevoice(audio_path, audio)
        except Exception as e:
            print(f"  [SAR] SenseVoice inference failed: {e}")
            return self._annotate_fallback(audio)

    def _annotate_sensevoice(self, audio_path: str, audio: np.ndarray) -> Dict[str, Any]:
        """使用SenseVoice进行属性识别"""
        # FunASR可以直接接受文件路径
        try:
            result = self.model.generate(
                input=audio_path,
                output_dir=None
            )
        except Exception as e:
            # 尝试使用wav tensor
            wav_tensor = torch.from_numpy(audio).unsqueeze(0).float()
            try:
                result = self.model.generate(
                    input=wav_tensor,
                    output_dir=None
                )
            except Exception as e2:
                print(f"  [SAR] generate failed: {e2}")
                raise e

        # 解析结果
        gender = "unknown"
        age = "unknown"
        confidence_gender = 0.5
        confidence_age = 0.5
        embedding = None

        if result and len(result) > 0:
            res = result[0]

            # 尝试解析属性
            # SenseVoice可能输出text格式的结果
            if 'text' in res:
                text = res['text']
                gender, age = self._parse_sensevoice_text(text)

            # 尝试获取embedding
            if 'embedding' in res:
                embedding = res['embedding']
            elif 'feats' in res:
                embedding = res['feats']

        # 如果没有embedding，提取一个简单的embedding
        if embedding is None:
            embedding = self._extract_simple_embedding(audio)

        predictions = {
            "attributes": {
                "gender": {
                    "label": gender,
                    "confidence": confidence_gender
                },
                "age": {
                    "label": age,
                    "confidence": confidence_age
                }
            },
            "speaker_embedding": {
                "vector": self._safe_float16(embedding.tolist()[:16] if hasattr(embedding, 'tolist') else embedding[:16]),
                "dimension": len(embedding) if hasattr(embedding, '__len__') else 192
            }
        }

        logits_dict = {
            "gender": gender,
            "age": age
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _annotate_fallback(self, audio: np.ndarray) -> Dict[str, Any]:
        """备用方案：使用简单编码器"""
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=80)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)

        # 提取embedding
        with torch.no_grad():
            embedding = self.model(mfcc_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        predictions = {
            "attributes": {
                "gender": {
                    "label": "unknown",
                    "confidence": 0.5
                },
                "age": {
                    "label": "unknown",
                    "confidence": 0.5
                }
            },
            "speaker_embedding": {
                "vector": self._safe_float16(embedding.tolist()[:16]),
                "dimension": len(embedding)
            }
        }

        return {
            "predictions": predictions,
            "logits": {}
        }

    def _parse_sensevoice_text(self, text: str) -> tuple:
        """解析SenseVoice输出文本中的属性信息"""
        gender = "unknown"
        age = "unknown"

        # SenseVoice可能输出类似: "性别:男 年龄:青年" 的格式
        text_lower = text.lower()

        # 性别识别
        if '女' in text or 'female' in text_lower or 'woman' in text_lower:
            gender = "female"
        elif '男' in text or 'male' in text_lower or 'man' in text_lower:
            gender = "male"

        # 年龄识别
        if '儿童' in text or 'child' in text_lower or 'kid' in text_lower:
            age = "child"
        elif '青年' in text or 'young' in text_lower:
            age = "young"
        elif '中年' in text or 'middle' in text_lower:
            age = "middle"
        elif '老年' in text or 'senior' in text_lower or 'old' in text_lower:
            age = "senior"

        return gender, age

    def _extract_simple_embedding(self, audio: np.ndarray) -> np.ndarray:
        """提取简单的说话人embedding"""
        # 使用MFCC统计作为简单embedding
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)
        # 统计特征
        mean = mfcc.mean(axis=1)
        std = mfcc.std(axis=1)
        # 拼接为192维
        embedding = np.concatenate([mean, std, mean[:32] * std[:32]])
        return embedding[:192]

    def _safe_float16(self, value):
        """安全转换为float16列表"""
        if isinstance(value, (list, np.ndarray)):
            result = []
            for v in value:
                if np.isnan(v) or np.isinf(v):
                    result.append(0.0)
                else:
                    result.append(float(v))
            return result

        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)


# 别名
SARAnnotator = SenseVoiceAttributeAnnotator
# 保持向后兼容
ECAPAAttributeAnnotator = SenseVoiceAttributeAnnotator
