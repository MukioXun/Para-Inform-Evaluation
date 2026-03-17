"""
音频预处理模块
负责音频的格式转换、切分、重采样等
"""
import os
import librosa
import soundfile as sf
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import uuid
import numpy as np


@dataclass
class AudioSegment:
    """音频片段数据类"""
    audio_id: str
    file_path: str
    start_time: float
    end_time: float
    duration: float
    sample_rate: int
    segment_path: Optional[str] = None
    waveform: Optional[np.ndarray] = None


class AudioProcessor:
    """音频预处理器"""

    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

    def __init__(
        self,
        sample_rate: int = 16000,
        max_segment_length: float = 30.0,
        output_dir: Optional[str] = None
    ):
        """
        初始化音频处理器

        Args:
            sample_rate: 目标采样率
            max_segment_length: 最大片段长度（秒）
            output_dir: 切分片段输出目录
        """
        self.sample_rate = sample_rate
        self.max_segment_length = max_segment_length
        self.output_dir = output_dir or "./data/intermediate/segments"

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            file_path: 音频文件路径

        Returns:
            waveform: 音频波形
            sr: 采样率
        """
        waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return waveform, sr

    def resample(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        重采样音频

        Args:
            waveform: 音频波形
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的波形
        """
        if orig_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        return waveform

    def segment_audio(
        self,
        waveform: np.ndarray,
        segment_length: float = 30.0,
        overlap: float = 0.0
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        将长音频切分为多个片段

        Args:
            waveform: 音频波形
            segment_length: 片段长度（秒）
            overlap: 片段重叠（秒）

        Returns:
            List of (segment_waveform, start_time, end_time)
        """
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step = segment_samples - overlap_samples

        segments = []
        for start in range(0, len(waveform), step):
            end = min(start + segment_samples, len(waveform))
            segment = waveform[start:end]
            start_time = start / self.sample_rate
            end_time = end / self.sample_rate
            segments.append((segment, start_time, end_time))

            if end >= len(waveform):
                break

        return segments

    def save_segment(
        self,
        waveform: np.ndarray,
        output_path: str
    ) -> str:
        """
        保存音频片段

        Args:
            waveform: 音频波形
            output_path: 输出路径

        Returns:
            保存的文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, waveform, self.sample_rate)
        return output_path

    def process_file(
        self,
        file_path: str,
        save_segments: bool = True
    ) -> List[AudioSegment]:
        """
        处理单个音频文件

        Args:
            file_path: 音频文件路径
            save_segments: 是否保存切分片段

        Returns:
            List of AudioSegment
        """
        # 加载音频
        waveform, sr = self.load_audio(file_path)
        duration = len(waveform) / sr

        audio_id = str(uuid.uuid4())[:8]

        # 如果音频较短，不切分
        if duration <= self.max_segment_length:
            segment = AudioSegment(
                audio_id=audio_id,
                file_path=file_path,
                start_time=0.0,
                end_time=duration,
                duration=duration,
                sample_rate=sr,
                waveform=waveform
            )
            return [segment]

        # 切分长音频
        segments_data = self.segment_audio(waveform, self.max_segment_length)
        segments = []

        for i, (seg_waveform, start, end) in enumerate(segments_data):
            seg_path = None
            if save_segments:
                seg_filename = f"{audio_id}_seg{i:03d}.wav"
                seg_path = os.path.join(self.output_dir, seg_filename)
                self.save_segment(seg_waveform, seg_path)

            segment = AudioSegment(
                audio_id=f"{audio_id}_{i:03d}",
                file_path=file_path,
                start_time=start,
                end_time=end,
                duration=end - start,
                sample_rate=sr,
                segment_path=seg_path,
                waveform=seg_waveform
            )
            segments.append(segment)

        return segments

    def process_directory(
        self,
        dir_path: str,
        save_segments: bool = True
    ) -> List[AudioSegment]:
        """
        处理目录下所有音频文件

        Args:
            dir_path: 目录路径
            save_segments: 是否保存切分片段

        Returns:
            List of AudioSegment
        """
        all_segments = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
                    file_path = os.path.join(root, file)
                    try:
                        segments = self.process_file(file_path, save_segments)
                        all_segments.extend(segments)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        return all_segments

    @staticmethod
    def get_audio_info(file_path: str) -> Dict:
        """
        获取音频文件信息

        Args:
            file_path: 音频文件路径

        Returns:
            音频信息字典
        """
        info = sf.info(file_path)
        return {
            "path": file_path,
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype
        }
