"""
Age Classifier - 年龄分类器
基于wav2vec2的年龄回归模型
输出: 年龄值 + 年龄段分类

注意: 使用与原始代码一致的AgeGenderModel结构
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, Any
from pathlib import Path
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from ..base_annotator import BaseAnnotator


class ModelHead(nn.Module):
    """分类/回归头"""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """年龄-性别联合预测模型 (与原始代码一致)"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


class AgeClassifier(BaseAnnotator):
    """年龄分类标注器"""

    TASK_NAME = "Age"

    # 年龄范围参数 (用于归一化反变换)
    # 假设模型在0-100岁范围内训练，输出是归一化值
    AGE_MIN = 0
    AGE_MAX = 100

    def load_model(self):
        """加载年龄分类模型"""
        model_path = self.config.get('model_path', '/home/u2023112559/qix/Models/Models/age-classification')

        print(f"  [Age] Loading model from: {model_path}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        # 使用正确的AgeGenderModel
        self.model = AgeGenderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"  [Age] Model loaded")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行年龄预测"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        audio = audio.astype(np.float32)

        # 预处理 (与原始代码一致)
        inputs = self.processor(audio, sampling_rate=self.sample_rate)
        input_values = inputs['input_values'][0]
        input_values = input_values.reshape(1, -1)
        input_tensor = torch.from_numpy(input_values).to(self.device)

        # 推理
        with torch.no_grad():
            hidden_states, logits_age, logits_gender = self.model(input_tensor)
            # logits_age是归一化的年龄值，需要反归一化
            normalized_age = logits_age[0, 0].item()

        # 年龄值可能是归一化的，尝试两种解释
        # 1. 直接作为年龄值
        direct_age = normalized_age

        # 2. 作为归一化值反归一化 (如果值在0-1范围内)
        if 0 <= normalized_age <= 1:
            actual_age = normalized_age * (self.AGE_MAX - self.AGE_MIN) + self.AGE_MIN
        else:
            actual_age = normalized_age

        # 使用合理的年龄值
        age_value = actual_age if actual_age > 1 else direct_age * 100

        # 年龄段分类
        age_group = self._classify_age_group(age_value)

        # 置信度计算
        confidence = self._compute_confidence(age_value)

        predictions = {
            "age_value": round(age_value, 2),
            "age_group": age_group,
            "confidence": confidence,
            "raw_output": round(normalized_age, 4)
        }

        logits_dict = {
            "normalized_age": round(normalized_age, 4),
            "estimated_age": round(age_value, 2)
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _classify_age_group(self, age: float) -> str:
        """年龄分组"""
        if age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 35:
            return "young_adult"
        elif age < 50:
            return "middle_aged"
        elif age < 65:
            return "senior"
        else:
            return "elderly"

    def _compute_confidence(self, age: float) -> float:
        """计算置信度 (基于年龄合理性)"""
        if 5 <= age <= 90:
            return 0.8
        elif 0 <= age <= 100:
            return 0.6
        else:
            return 0.3
