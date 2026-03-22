"""
SED标注器 - PANNs
声学事件检测，检测527类AudioSet事件
输出压缩格式：仅保留top_events和prob_summary
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..base_annotator import BaseAnnotator
from ...config.model_config import AUDIOSET_LABELS


class PANNsDetector(BaseAnnotator):
    """PANNs声学事件检测器"""

    TASK_NAME = "SED"

    # 重点关注的声学事件类别
    FOCUS_EVENTS = [
        "Speech",
        "Breathing",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Shout",
        "Screaming",
        "Whispering",
        "Laughter",
        "Crying, sobbing",
        "Singing",
        "Cough",
        "Sneeze",
        "Snoring",
    ]

    def load_model(self):
        """加载PANNs模型"""
        from torch import nn
        import torch.nn.functional as F

        # PANNs CNN14 模型定义（匹配官方结构）
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 不使用bias，与官方PANNs一致
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)

            def forward(self, x, pool_size=(2, 2)):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.max_pool2d(x, kernel_size=pool_size)
                return x

        class Cnn14(nn.Module):
            def __init__(self, sample_rate=32000, window_size=1024, hop_size=320,
                         mel_bins=64, fmin=50, fmax=14000, classes_num=527):
                super().__init__()
                self.bn0 = nn.BatchNorm2d(mel_bins)

                self.conv_block1 = ConvBlock(1, 64)
                self.conv_block2 = ConvBlock(64, 128)
                self.conv_block3 = ConvBlock(128, 256)
                self.conv_block4 = ConvBlock(256, 512)
                self.conv_block5 = ConvBlock(512, 1024)
                self.conv_block6 = ConvBlock(1024, 2048)

                self.fc1 = nn.Linear(2048, 2048)
                self.fc_audioset = nn.Linear(2048, classes_num)

            def forward(self, x):
                # x: [B, 1, mel_bins, T]
                x = x.transpose(1, 2)  # [B, mel_bins, 1, T]
                x = self.bn0(x)
                x = x.transpose(1, 2)  # [B, 1, mel_bins, T]

                x = self.conv_block1(x, (2, 2))
                x = self.conv_block2(x, (2, 2))
                x = self.conv_block3(x, (2, 2))
                x = self.conv_block4(x, (2, 2))
                x = self.conv_block5(x, (2, 2))
                x = self.conv_block6(x, (1, 1))

                x = torch.mean(x, dim=3)  # [B, 2048, T]
                x = torch.mean(x, dim=2)  # [B, 2048]

                x = F.dropout(x, p=0.5)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, p=0.5)
                output = self.fc_audioset(x)
                return output

        model_path = self.config.get('model_path')

        print(f"  [SED] Loading PANNs from: {model_path}")

        # 创建模型
        self.model = Cnn14()

        # 加载权重
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 过滤掉不需要的键
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # 跳过spectrogram_extractor和logmel_extractor
                if k.startswith('spectrogram_extractor') or k.startswith('logmel_extractor'):
                    continue
                filtered_state_dict[k] = v

            # 使用strict=False忽略不匹配的键
            missing, unexpected = self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"  [SED] PANNs weights loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")
        else:
            print(f"  [SED] Warning: Model weights not found at {model_path}")

        self.model.to(self.device)
        self.model.eval()

        # 加载标签
        self.labels = AUDIOSET_LABELS

        print(f"  [SED] PANNs ready (527 classes)")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """
        执行声学事件检测
        输出压缩格式：仅保留top_events和prob_summary，剔除527维全量logits
        """
        # 加载音频 (PANNs使用32kHz)
        audio, sr = librosa.load(audio_path, sr=32000)

        # 提取Mel谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=32000,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            fmin=50,
            fmax=14000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 转换为模型输入 [B, 1, mel_bins, T]
        x = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        # 推理
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # 阈值
        threshold = self.config.get('threshold', 0.5)

        # === top_events: 仅保留置信度>=阈值的事件 ===
        top_indices = np.argsort(probs)[::-1][:20]  # 取top 20候选

        top_events = []
        for idx in top_indices:
            event_name = self._get_label(idx)
            prob = float(probs[idx])
            if prob >= threshold:
                top_events.append({
                    "event": event_name,
                    "confidence": round(prob, 4)
                })

        # === prob_summary: 重点事件的概率摘要 ===
        focus_probs = {}
        for event in self.FOCUS_EVENTS:
            idx = self._get_event_index(event)
            if idx is not None:
                focus_probs[event] = round(float(probs[idx]), 4)

        # 统计摘要
        predictions = {
            "top_events": top_events[:10],  # 最多保留10个
            "prob_summary": {
                "focus_events": focus_probs,
                "speech_detected": focus_probs.get("Speech", 0) >= threshold,
                "breathing_detected": focus_probs.get("Breathing", 0) >= threshold,
                "child_speech_detected": focus_probs.get("Child speech, kid speaking", 0) >= threshold,
            },
            "primary_event": top_events[0]["event"] if top_events else "unknown",
            "primary_confidence": top_events[0]["confidence"] if top_events else 0.0
        }

        # logits不再保存全量527维，只保存摘要信息
        logits_dict = {
            "num_detected_events": len(top_events),
            "max_probability": round(float(np.max(probs)), 4),
            "mean_probability": round(float(np.mean(probs)), 4)
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _get_event_index(self, event_name: str) -> Optional[int]:
        """根据事件名称获取索引"""
        try:
            return self.labels.index(event_name)
        except ValueError:
            return None

    def _get_label(self, idx: int) -> str:
        """获取AudioSet标签"""
        if idx < len(self.labels):
            return self.labels[idx]
        return f"class_{idx}"


# 别名
SEDAnnotator = PANNsDetector
