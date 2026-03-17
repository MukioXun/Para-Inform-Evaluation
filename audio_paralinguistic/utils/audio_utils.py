"""
音频处理工具函数
"""
import os
from typing import Tuple, Optional, Union
import numpy as np


def load_audio(
    file_path: str,
    sample_rate: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件

    Args:
        file_path: 音频文件路径
        sample_rate: 目标采样率（None表示保持原采样率）
        mono: 是否转为单声道

    Returns:
        (waveform, sr): 音频波形和采样率
    """
    import librosa
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=mono)
    return waveform, sr


def get_audio_duration(file_path: str) -> float:
    """
    获取音频时长

    Args:
        file_path: 音频文件路径

    Returns:
        时长（秒）
    """
    import soundfile as sf
    info = sf.info(file_path)
    return info.duration


def resample_audio(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    重采样音频

    Args:
        waveform: 音频波形
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的波形
    """
    import librosa
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return waveform


def split_audio_by_silence(
    waveform: np.ndarray,
    sr: int,
    min_silence_duration: float = 0.5,
    silence_threshold: float = -40
) -> list:
    """
    按静音分割音频

    Args:
        waveform: 音频波形
        sr: 采样率
        min_silence_duration: 最小静音时长（秒）
        silence_threshold: 静音阈值（dB）

    Returns:
        分割后的音频片段列表
    """
    import librosa

    # 计算能量
    rms = librosa.feature.rms(y=waveform)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # 找到静音区域
    silence_mask = rms_db < silence_threshold

    # 找到分割点
    min_silence_samples = int(min_silence_duration * sr / 512)  # 512是hop_length

    segments = []
    start = 0

    silence_count = 0
    for i, is_silence in enumerate(silence_mask):
        if is_silence:
            silence_count += 1
        else:
            if silence_count >= min_silence_samples:
                end = i * 512
                if end > start:
                    segments.append(waveform[start:end])
                start = i * 512
            silence_count = 0

    # 添加最后一段
    if start < len(waveform):
        segments.append(waveform[start:])

    return segments


def compute_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    计算频谱图

    Args:
        waveform: 音频波形
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 跳跃长度

    Returns:
        频谱图
    """
    import librosa
    spec = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    return spec_db


def extract_mfcc(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    提取MFCC特征

    Args:
        waveform: 音频波形
        sr: 采样率
        n_mfcc: MFCC系数数量

    Returns:
        MFCC特征
    """
    import librosa
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfcc
