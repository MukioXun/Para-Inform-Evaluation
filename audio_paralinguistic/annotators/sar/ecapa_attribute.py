"""
SAR标注器 - ECAPA-TDNN
说话人属性识别，提取embedding和属性预测
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..base_annotator import BaseAnnotator


class ECAPAAttributeAnnotator(BaseAnnotator):
    """ECAPA-TDNN说话人属性标注器"""

    TASK_NAME = "SAR"

    def load_model(self):
        """加载ECAPA-TDNN模型"""
        import torch.nn as nn
        import torch.nn.functional as F

        # 简化的ECAPA-TDNN encoder
        class ECAPAEncoder(nn.Module):
            """简化的ECAPA-TDNN编码器"""
            def __init__(self, input_dim=80, hidden_dim=512, embedding_dim=192):
                super().__init__()
                self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
                self.bn3 = nn.BatchNorm1d(hidden_dim)
                self.fc = nn.Linear(hidden_dim, embedding_dim)

            def forward(self, x):
                # x: [B, C, T]
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                x = torch.mean(x, dim=2)  # Temporal pooling
                x = self.fc(x)
                return x

        model_path = self.config.get('model_path')

        print(f"  [SAR] Loading ECAPA-TDNN from: {model_path}")

        # 创建模型
        self.model = ECAPAEncoder()

        # 尝试加载权重
        if model_path and Path(model_path).exists():
            try:
                # 尝试加载embedding_model.ckpt
                ckpt_path = Path(model_path) / "embedding_model.ckpt"
                if ckpt_path.exists():
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                    print(f"  [SAR] Loaded weights from {ckpt_path}")
            except Exception as e:
                print(f"  [SAR] Warning: Could not load weights: {e}")

        self.model.to(self.device)
        self.model.eval()

        print(f"  [SAR] ECAPA-TDNN initialized (embedding_dim=192)")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行说话人属性识别"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=80)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)  # [1, 80, T]

        # 提取embedding
        with torch.no_grad():
            embedding = self.model(mfcc_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        # 属性预测（基于embedding的简单分类）
        # 注意：实际需要训练好的属性分类头
        gender_pred = self._predict_gender(embedding)
        age_pred = self._predict_age(embedding)

        predictions = {
            "attributes": {
                "gender": {
                    "label": gender_pred["label"],
                    "confidence": gender_pred["confidence"]
                },
                "age_group": {
                    "label": age_pred["label"],
                    "confidence": age_pred["confidence"]
                }
            },
            "speaker_embedding": {
                "vector": embedding.tolist()[:10],  # 只保存前10维用于调试
                "dimension": len(embedding)
            }
        }

        logits_dict = {
            "gender_logits": gender_pred["logits"],
            "age_logits": age_pred["logits"]
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _predict_gender(self, embedding: np.ndarray) -> dict:
        """预测性别（简化版，实际需要分类头）"""
        # 基于embedding的简单启发式
        # 实际应该用训练好的分类器
        return {
            "label": "unknown",
            "confidence": 0.5,
            "logits": [0.33, 0.33, 0.34]
        }

    def _predict_age(self, embedding: np.ndarray) -> dict:
        """预测年龄组（简化版）"""
        return {
            "label": "unknown",
            "confidence": 0.5,
            "logits": [0.2, 0.2, 0.2, 0.2, 0.2]
        }


# 别名
SARAnnotator = ECAPAAttributeAnnotator
