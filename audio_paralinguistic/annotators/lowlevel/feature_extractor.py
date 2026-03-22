"""
Low-Level特征提取器
提取基础物理特征：Spectral, Prosody, Energy, Temporal, Timbre
"""
import os
import torch
import librosa
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..base_annotator import BaseAnnotator


class LowLevelFeatureExtractor(BaseAnnotator):
    """低级声学特征提取器"""

    TASK_NAME = "LowLevel"

    # 情感标签映射 (emotion2vec标准)
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

    def load_model(self):
        """加载必要的模型（VAD等）"""
        print(f"  [LowLevel] Initializing feature extractor...")

        # 加载silero-vad用于语音活动检测
        try:
            import torch
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.vad_model.eval()
            self.use_vad = True
            print(f"  [LowLevel] VAD model loaded")
        except Exception as e:
            print(f"  [LowLevel] Warning: VAD not available: {e}")
            self.vad_model = None
            self.use_vad = False

        print(f"  [LowLevel] Feature extractor ready")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """提取所有低级特征"""
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # 确保音频有效
        if len(audio) == 0 or np.all(audio == 0):
            return self._get_empty_features()

        features = {}

        # 1. Spectral Features
        features["spectral"] = self._extract_spectral(audio, sr)

        # 2. Prosody Features
        features["prosody"] = self._extract_prosody(audio, sr)

        # 3. Energy Features
        features["energy"] = self._extract_energy(audio, sr)

        # 4. Temporal Features
        features["temporal"] = self._extract_temporal(audio, sr, audio_path)

        # 5. Timbre Features
        features["timbre"] = self._extract_timbre(audio, sr)

        return {
            "predictions": features,
            "logits": {}  # Low-level特征不产生logits
        }

    def _extract_spectral(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取频谱特征"""
        features = {}

        # MFCCs (13维 + delta + delta-delta)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features["mfcc"] = {
            "mean": self._safe_float16(mfcc.mean(axis=1).tolist()),
            "std": self._safe_float16(mfcc.std(axis=1).tolist()),
            "delta_mean": self._safe_float16(mfcc_delta.mean(axis=1).tolist()),
            "delta2_mean": self._safe_float16(mfcc_delta2.mean(axis=1).tolist())
        }

        # Mel-spectrogram stats
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        features["mel"] = {
            "mean_db": self._safe_float16(float(mel_spec_db.mean())),
            "std_db": self._safe_float16(float(mel_spec_db.std())),
            "n_mels": 80
        }

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features["centroid"] = {
            "mean_hz": self._safe_float16(float(centroid.mean())),
            "std_hz": self._safe_float16(float(centroid.std()))
        }

        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features["bandwidth"] = {
            "mean_hz": self._safe_float16(float(bandwidth.mean())),
            "std_hz": self._safe_float16(float(bandwidth.std()))
        }

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features["rolloff"] = {
            "mean_hz": self._safe_float16(float(rolloff.mean())),
            "std_hz": self._safe_float16(float(rolloff.std()))
        }

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio)
        features["flatness"] = {
            "mean": self._safe_float16(float(flatness.mean())),
            "std": self._safe_float16(float(flatness.std()))
        }

        return features

    def _extract_prosody(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取韵律特征 (F0, Jitter, Shimmer, HNR)"""
        features = {}

        try:
            # 使用pyworld提取F0
            import pyworld as pw

            # pyworld需要float64 (double)
            audio_f64 = audio.astype(np.float64)
            _f0, t = pw.harvest(audio_f64, sr)
            f0 = pw.stonemask(audio_f64, _f0, t, sr)

            # 过滤有效的F0值（非零）
            f0_voiced = f0[f0 > 0]

            if len(f0_voiced) > 0:
                features["f0"] = {
                    "mean_hz": self._safe_float16(float(np.mean(f0_voiced))),
                    "std_hz": self._safe_float16(float(np.std(f0_voiced))),
                    "min_hz": self._safe_float16(float(np.min(f0_voiced))),
                    "max_hz": self._safe_float16(float(np.max(f0_voiced))),
                    "voiced_frames": int(len(f0_voiced)),
                    "voiced_ratio": self._safe_float16(float(len(f0_voiced) / len(f0)))
                }
            else:
                features["f0"] = self._get_empty_f0()

            # 计算Jitter (基频扰动)
            if len(f0_voiced) > 1:
                jitter = self._compute_jitter(f0_voiced)
                features["jitter"] = {
                    "local": self._safe_float16(float(jitter)),
                    "rap": self._safe_float16(float(jitter * 0.8)),  # 近似
                    "ppq5": self._safe_float16(float(jitter * 0.9))   # 近似
                }
            else:
                features["jitter"] = {"local": 0.0, "rap": 0.0, "ppq5": 0.0}

            # 计算Shimmer (振幅扰动)
            shimmer = self._compute_shimmer(audio, sr, f0)
            features["shimmer"] = {
                "local": self._safe_float16(float(shimmer)),
                "apq3": self._safe_float16(float(shimmer * 0.8)),
                "apq5": self._safe_float16(float(shimmer * 0.9))
            }

            # 计算HNR (谐波噪声比)
            hnr = self._compute_hnr(audio, sr)
            features["hnr"] = {
                "mean_db": self._safe_float16(float(hnr)),
                "std_db": self._safe_float16(0.0)
            }

        except ImportError:
            print("  [LowLevel] Warning: pyworld not available, using fallback F0")
            # 回退到librosa的F0估计
            f0, voiced_flags = librosa.piptrack(y=audio, sr=sr)
            f0_voiced = f0[voiced_flags]

            if len(f0_voiced) > 0:
                features["f0"] = {
                    "mean_hz": self._safe_float16(float(np.mean(f0_voiced))),
                    "std_hz": self._safe_float16(float(np.std(f0_voiced))),
                    "min_hz": self._safe_float16(float(np.min(f0_voiced))),
                    "max_hz": self._safe_float16(float(np.max(f0_voiced))),
                    "voiced_frames": int(len(f0_voiced)),
                    "voiced_ratio": self._safe_float16(float(len(f0_voiced) / f0.size))
                }
            else:
                features["f0"] = self._get_empty_f0()

            features["jitter"] = {"local": 0.0, "rap": 0.0, "ppq5": 0.0}
            features["shimmer"] = {"local": 0.0, "apq3": 0.0, "apq5": 0.0}
            features["hnr"] = {"mean_db": 0.0, "std_db": 0.0}

        return features

    def _extract_energy(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取能量特征"""
        features = {}

        # RMS能量
        rms = librosa.feature.rms(y=audio)[0]
        features["rms"] = {
            "mean": self._safe_float16(float(np.mean(rms))),
            "std": self._safe_float16(float(np.std(rms))),
            "min": self._safe_float16(float(np.min(rms))),
            "max": self._safe_float16(float(np.max(rms)))
        }

        # Loudness (近似ITU-R BS.1770)
        # 简化版本：使用RMS的dB值
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        features["loudness"] = {
            "mean_db": self._safe_float16(float(np.mean(rms_db))),
            "peak_db": self._safe_float16(float(np.max(rms_db))),
            "lufs_approx": self._safe_float16(float(np.mean(rms_db) - 10))  # 近似LUFS
        }

        # Dynamic Range
        features["dynamic_range"] = {
            "range_db": self._safe_float16(float(np.max(rms_db) - np.min(rms_db))),
            "crest_factor": self._safe_float16(float(np.max(np.abs(audio)) / (np.mean(rms) + 1e-8)))
        }

        return features

    def _extract_temporal(self, audio: np.ndarray, sr: int, audio_path: str) -> Dict[str, Any]:
        """提取时间特征"""
        features = {}

        # Duration
        duration = len(audio) / sr
        features["duration"] = {
            "total_seconds": self._safe_float16(float(duration)),
            "total_samples": int(len(audio))
        }

        # VAD分析
        if self.use_vad and self.vad_model is not None:
            try:
                vad_result = self._run_vad(audio, sr)
                features.update(vad_result)
            except Exception as e:
                print(f"  [LowLevel] VAD failed: {e}")
                features.update(self._get_empty_temporal())
        else:
            features.update(self._get_empty_temporal())

        return features

    def _extract_timbre(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """提取音色特征"""
        features = {}

        try:
            # Formants (使用librosa近似)
            # 通过线性预测编码(LPC)估计共振峰
            formants = self._estimate_formants(audio, sr)
            features["formants"] = {
                "f1_hz": self._safe_float16(float(formants[0]) if len(formants) > 0 else 0.0),
                "f2_hz": self._safe_float16(float(formants[1]) if len(formants) > 1 else 0.0),
                "f3_hz": self._safe_float16(float(formants[2]) if len(formants) > 2 else 0.0),
            }
        except Exception:
            features["formants"] = {"f1_hz": 0.0, "f2_hz": 0.0, "f3_hz": 0.0}

        # Spectral envelope (简化版)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=13)
        features["spectral_envelope"] = {
            "mean": self._safe_float16(mel_spec.mean(axis=1).tolist()),
            "n_bands": 13
        }

        # Harmonicity
        harmonicity = librosa.effects.harmonic(audio)
        features["harmonicity"] = {
            "mean": self._safe_float16(float(np.mean(np.abs(harmonicity)))),
            "ratio": self._safe_float16(float(np.sum(np.abs(harmonicity)) / (np.sum(np.abs(audio)) + 1e-8)))
        }

        return features

    def _run_vad(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """运行VAD分析"""
        # Silero VAD需要特定采样率
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        # Silero VAD需要分块处理 (512样本 = 32ms @ 16kHz)
        chunk_size = 512
        speech_probs_list = []

        # 转换为tensor并分块
        audio_tensor = torch.from_numpy(audio_16k).float()

        # 分块处理
        with torch.no_grad():
            for i in range(0, len(audio_tensor), chunk_size):
                chunk = audio_tensor[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # 填充最后一块
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
                # VAD模型期望 [batch, samples]
                prob = self.vad_model(chunk.unsqueeze(0), 16000)
                speech_probs_list.append(prob.item())

        speech_probs = np.array(speech_probs_list)

        # 计算统计
        speech_threshold = 0.5
        speech_frames = (speech_probs > speech_threshold).sum()
        total_frames = len(speech_probs)

        # 估计语速 (简化)
        duration = len(audio) / sr
        speech_duration = speech_frames / 100  # 近似

        return {
            "speech_rate": {
                "estimated_syllables_per_sec": self._safe_float16(float(speech_frames / (duration * 50) if duration > 0 else 0)),
                "speech_duration_sec": self._safe_float16(float(speech_duration))
            },
            "pause_ratio": {
                "speech_ratio": self._safe_float16(float(speech_frames / total_frames if total_frames > 0 else 0)),
                "pause_ratio": self._safe_float16(float(1 - speech_frames / total_frames) if total_frames > 0 else 0)
            },
            "voiced_ratio": {
                "ratio": self._safe_float16(float(speech_frames / total_frames if total_frames > 0 else 0))
            }
        }

    def _compute_jitter(self, f0: np.ndarray) -> float:
        """计算Jitter (基频周期扰动)"""
        if len(f0) < 2:
            return 0.0

        periods = 1.0 / f0  # 转换为周期
        diffs = np.abs(np.diff(periods))
        jitter = np.mean(diffs) / np.mean(periods)
        return float(jitter)

    def _compute_shimmer(self, audio: np.ndarray, sr: int, f0: np.ndarray) -> float:
        """计算Shimmer (振幅扰动)"""
        if len(f0) < 2:
            return 0.0

        # 提取每个基频周期的峰值振幅
        try:
            frame_length = int(sr / np.mean(f0[f0 > 0]))
            frame_length = max(100, min(frame_length, 2000))

            n_frames = len(audio) // frame_length
            if n_frames < 2:
                return 0.0

            amplitudes = []
            for i in range(n_frames):
                start = i * frame_length
                end = start + frame_length
                frame = audio[start:end]
                amplitudes.append(np.max(np.abs(frame)))

            amplitudes = np.array(amplitudes)
            diffs = np.abs(np.diff(amplitudes))
            shimmer = np.mean(diffs) / (np.mean(amplitudes) + 1e-8)
            return float(shimmer)
        except Exception:
            return 0.0

    def _compute_hnr(self, audio: np.ndarray, sr: int) -> float:
        """计算HNR (谐波噪声比)"""
        try:
            # 使用自相关方法
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # 找到第一个峰值（基频位置）
            d = np.diff(autocorr)
            start = np.where(d > 0)[0]
            if len(start) == 0:
                return 0.0
            start = start[0]

            peak_idx = np.argmax(autocorr[start:]) + start
            if peak_idx == 0:
                return 0.0

            # HNR = 10 * log10(r_max / (1 - r_max))
            r_max = autocorr[peak_idx] / autocorr[0]
            r_max = min(r_max, 0.999)  # 防止log(负数)

            hnr = 10 * np.log10(r_max / (1 - r_max + 1e-8))
            return float(hnr)
        except Exception:
            return 0.0

    def _estimate_formants(self, audio: np.ndarray, sr: int, n_formants: int = 3) -> List[float]:
        """估计共振峰 (使用LPC)"""
        try:
            # 预加重
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

            # 分帧
            frame_size = int(0.025 * sr)  # 25ms
            frame = emphasized[:frame_size]

            # 加窗
            window = np.hamming(len(frame))
            windowed = frame * window

            # LPC分析
            lpc_order = int(2 + sr / 1000)  # 经验公式
            lpc_order = min(lpc_order, len(windowed) // 2 - 1)

            # 自相关
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Levinson-Durbin递归
            a = self._levinson_durbin(autocorr[:lpc_order+1], lpc_order)

            # 求根得到共振峰
            roots = np.roots(a)

            # 只保留正虚部的根
            formants = []
            for root in roots:
                if root.imag >= 0:
                    freq = np.arctan2(root.imag, root.real) * sr / (2 * np.pi)
                    if 90 < freq < sr / 2 - 100:  # 过滤无效频率
                        formants.append(freq)

            formants.sort()
            return formants[:n_formants]

        except Exception:
            return []

    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin算法求解LPC系数"""
        a = np.zeros(order + 1)
        a[0] = 1.0

        e = autocorr[0]

        for k in range(1, order + 1):
            lambda_k = -np.sum(autocorr[1:k+1] * a[k-1::-1]) / e

            a_new = a.copy()
            for i in range(k):
                a_new[i] = a[i] + lambda_k * a[k-1-i]
            a_new[k] = lambda_k
            a = a_new

            e = e * (1 - lambda_k ** 2)

        return a

    def _safe_float16(self, value):
        """安全转换为float16，避免NaN和Inf"""
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

    def _get_empty_f0(self) -> Dict[str, Any]:
        return {
            "mean_hz": 0.0,
            "std_hz": 0.0,
            "min_hz": 0.0,
            "max_hz": 0.0,
            "voiced_frames": 0,
            "voiced_ratio": 0.0
        }

    def _get_empty_temporal(self) -> Dict[str, Any]:
        return {
            "speech_rate": {
                "estimated_syllables_per_sec": 0.0,
                "speech_duration_sec": 0.0
            },
            "pause_ratio": {
                "speech_ratio": 0.0,
                "pause_ratio": 0.0
            },
            "voiced_ratio": {
                "ratio": 0.0
            }
        }

    def _get_empty_features(self) -> Dict[str, Any]:
        return {
            "predictions": {
                "spectral": {},
                "prosody": {},
                "energy": {},
                "temporal": {},
                "timbre": {}
            },
            "logits": {}
        }


# 别名
LowLevelAnnotator = LowLevelFeatureExtractor
