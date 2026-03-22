"""
Embedding提取器
提取深度模型中层表示：wav2vec2, HuBERT, CLAP
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..base_annotator import BaseAnnotator


class EmbeddingExtractor(BaseAnnotator):
    """深度模型Embedding提取器"""

    TASK_NAME = "Embeddings"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = {}
        self.processors = {}

    def load_model(self):
        """加载所有embedding模型"""
        print(f"  [Embeddings] Loading embedding models...")

        # 加载wav2vec2
        self._load_wav2vec2()

        # 加载HuBERT
        self._load_hubert()

        # 加载CLAP
        self._load_clap()

        print(f"  [Embeddings] All models loaded")

    def _load_wav2vec2(self):
        """加载wav2vec2-base-960h"""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            model_path = self.config.get(
                'wav2vec2_path',
                'facebook/wav2vec2-base-960h'
            )

            print(f"    Loading wav2vec2 from: {model_path}")

            self.processors['wav2vec2'] = Wav2Vec2Processor.from_pretrained(model_path)
            self.models['wav2vec2'] = Wav2Vec2Model.from_pretrained(model_path)
            self.models['wav2vec2'].to(self.device)
            self.models['wav2vec2'].eval()

            print(f"    wav2vec2 loaded (dim=768)")

        except Exception as e:
            print(f"    Warning: Failed to load wav2vec2: {e}")

    def _load_hubert(self):
        """加载HuBERT-base-ls960"""
        try:
            from transformers import HubertModel, Wav2Vec2Processor

            model_path = self.config.get(
                'hubert_path',
                'facebook/hubert-base-ls960'
            )

            print(f"    Loading HuBERT from: {model_path}")

            self.processors['hubert'] = Wav2Vec2Processor.from_pretrained(model_path)
            self.models['hubert'] = HubertModel.from_pretrained(model_path)
            self.models['hubert'].to(self.device)
            self.models['hubert'].eval()

            print(f"    HuBERT loaded (dim=768)")

        except Exception as e:
            print(f"    Warning: Failed to load HuBERT: {e}")

    def _load_clap(self):
        """加载CLAP音频编码器"""
        try:
            from transformers import ClapModel, ClapProcessor

            model_path = self.config.get(
                'clap_path',
                'laion/clap-htsat-unfused'
            )

            print(f"    Loading CLAP from: {model_path}")

            self.processors['clap'] = ClapProcessor.from_pretrained(model_path)
            self.models['clap'] = ClapModel.from_pretrained(model_path)
            self.models['clap'].to(self.device)
            self.models['clap'].eval()

            print(f"    CLAP loaded (dim=512)")

        except Exception as e:
            print(f"    Warning: Failed to load CLAP: {e}")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """提取所有embedding"""
        # 加载音频 (统一16kHz)
        audio, sr = librosa.load(audio_path, sr=16000)

        embeddings = {}

        # wav2vec2 embedding
        if 'wav2vec2' in self.models:
            embeddings['wav2vec2'] = self._extract_wav2vec2(audio)

        # HuBERT embedding
        if 'hubert' in self.models:
            embeddings['hubert'] = self._extract_hubert(audio)

        # CLAP embedding
        if 'clap' in self.models:
            embeddings['clap'] = self._extract_clap(audio, sr)

        return {
            "predictions": embeddings,
            "logits": {}
        }

    def _extract_wav2vec2(self, audio: np.ndarray) -> Dict[str, Any]:
        """提取wav2vec2 embedding (utterance-level, mean pooling)"""
        try:
            processor = self.processors['wav2vec2']
            model = self.models['wav2vec2']

            # 预处理
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 提取hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state  # [B, T, 768]

            # Mean pooling -> utterance-level embedding
            utterance_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

            return {
                "vector": self._safe_float16(utterance_embedding.tolist()),
                "dimension": int(utterance_embedding.shape[0]),
                "pooling_method": "mean",
                "model": "wav2vec2-base-960h"
            }

        except Exception as e:
            print(f"    wav2vec2 extraction failed: {e}")
            return {"vector": [], "dimension": 0, "error": str(e)}

    def _extract_hubert(self, audio: np.ndarray) -> Dict[str, Any]:
        """提取HuBERT embedding (utterance-level, mean pooling)"""
        try:
            processor = self.processors['hubert']
            model = self.models['hubert']

            # 预处理
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 提取hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state  # [B, T, 768]

            # Mean pooling
            utterance_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()

            return {
                "vector": self._safe_float16(utterance_embedding.tolist()),
                "dimension": int(utterance_embedding.shape[0]),
                "pooling_method": "mean",
                "model": "hubert-base-ls960"
            }

        except Exception as e:
            print(f"    HuBERT extraction failed: {e}")
            return {"vector": [], "dimension": 0, "error": str(e)}

    def _extract_clap(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取CLAP音频embedding"""
        try:
            processor = self.processors['clap']
            model = self.models['clap']

            # 预处理
            inputs = processor(
                audios=audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 提取audio embedding
            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs)
                # audio_embed: [B, 512]

            embedding = audio_embed.squeeze().cpu().numpy()

            return {
                "vector": self._safe_float16(embedding.tolist()),
                "dimension": int(embedding.shape[0]),
                "model": "clap-htsat-unfused"
            }

        except Exception as e:
            print(f"    CLAP extraction failed: {e}")
            return {"vector": [], "dimension": 0, "error": str(e)}

    def _safe_float16(self, value):
        """安全转换为float16列表，避免NaN和Inf"""
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
EmbeddingAnnotator = EmbeddingExtractor
