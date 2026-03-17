"""
模型配置模块
定义各模型的路径、API配置等
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import os


@dataclass
class APIConfig:
    """API配置类"""
    api_key: str = ""
    endpoint: str = ""
    max_retry: int = 5
    retry_sleep: float = 0.5
    timeout: int = 60

    # 千帆大模型配置
    qwen_api_key: str = os.getenv("QWEN_API_KEY", "")
    qwen_endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"

    # Google配置
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")


@dataclass
class ModelConfig:
    """模型配置类"""

    # ============ 语义维度模型 ============
    # ASR模型
    asr_model_name: str = "iic/SenseVoiceSmall"
    asr_model_device: str = "cuda"

    # Faster-Whisper配置
    faster_whisper_model: str = "tiny"
    faster_whisper_device: str = "cuda"

    # 意图识别模型
    intent_model_name: str = "microsoft/Phi-4-mini-instruct"

    # ============ 声学维度模型 ============
    # 说话人识别模型
    speaker_model_name: str = "cam++"
    speaker_model_path: Optional[str] = None

    # 情感识别模型
    emotion_model_name: str = "emotion2vec_plus_large"
    emotion_model_device: str = "cuda"

    # 副语言特征模型
    paralingual_model_name: str = "facebook/w2v-bert-2.0"

    # 多维情感模型 (VAD)
    vad_model_name: str = "MERaLiON-SER"

    # ============ 通用配置 ============
    device: str = "cuda"
    sample_rate: int = 16000
    max_audio_length: float = 30.0  # 秒

    # 并行配置
    num_workers: int = 4

    def get_model_path(self, model_name: str) -> str:
        """获取模型路径"""
        # 可扩展：从环境变量或配置文件读取自定义路径
        return model_name
