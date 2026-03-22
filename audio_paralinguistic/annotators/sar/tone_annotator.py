"""
Tone Annotator - 语气标注器
基于Audio-Reasoner (Qwen2-Audio) 的语气识别
输出: 仅保留<CAPTION>内容</CAPTION>中的语气描述

运行环境要求:
- conda activate Audio-Reasoner
- 或安装: pip install swift
"""
import re
from typing import Dict, Any
from pathlib import Path

from ..base_annotator import BaseAnnotator


class ToneAnnotator(BaseAnnotator):
    """语气标注器 - 使用Audio-Reasoner"""

    TASK_NAME = "Tone"

    # 语气识别专用prompt
#     TONE_PROMPT = """请分析这段语音中说话人的语气特点。

# 重要提示：
# 1. 请完全忽略语音中说的具体文字内容
# 2. 只关注说话人的语气、语调、情感色彩
# 3. 不要描述语音内容，只描述语气特征

# 请从以下维度分析语气：
# - 情感基调（如：严肃、轻松、激动、平静等）
# - 语速特点（如：急促、缓慢、适中、有变化等）
# - 语调特征（如：上扬、下沉、平稳、波动等）
# - 声音能量（如：有力、柔和、虚弱等）
# - 情绪状态（如：自信、犹豫、焦虑、坦然等）

# 请用简洁的语言总结这段语音的语气特点。"""
    TONE_PROMPT = """Analyze the speaker's vocal delivery in this audio while adhering to the following strict constraints:
    Zero Semantic Content: Treat the audio as if it were in a language you do not understand. Do not transcribe or summarize the verbal content.
    Focus on Prosody: Analyze only the non-verbal cues and acoustic features.

    Dimensions for Analysis:
    Affective Base: The underlying emotional mood (e.g., solemn, lighthearted, agitated).
    Speech Rate: Tempo, rhythmic regularity, and use of pauses.
    Inflection & Pitch: Pitch range, contours (rising/falling), and melodic variation.
    Vocal Intensity: Breath support, projection, and dynamic range (strong vs. frail).
    Psychological Profile: Perceived confidence, uncertainty, urgency, or composure.

    Output: Provide a precise and professional summary of these vocal traits."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = None
        self._swift_available = None

    def _check_swift(self) -> bool:
        """检查swift是否可用"""
        if self._swift_available is not None:
            return self._swift_available

        try:
            from swift.llm import PtEngine
            self._swift_available = True
        except ImportError:
            self._swift_available = False
            print(f"  [Tone] Warning: swift module not found!")
            print(f"  [Tone] Please run in Audio-Reasoner environment:")
            print(f"  [Tone]   conda activate Audio-Reasoner")
            print(f"  [Tone] Or install: pip install swift")

        return self._swift_available

    def load_model(self):
        """加载Audio-Reasoner模型"""
        if not self._check_swift():
            print(f"  [Tone] Model loading skipped (swift not available)")
            return

        from swift.llm import PtEngine

        model_path = self.config.get('model_path', '/home/u2023112559/qix/Models/Models/Audio-Reasoner')

        print(f"  [Tone] Loading Audio-Reasoner from: {model_path}")

        try:
            # Audio-Reasoner使用swift的PtEngine
            self.engine = PtEngine(
                model_path,
                max_batch_size=self.config.get('batch_size', 1),
                model_type='qwen2_audio'
            )
            print(f"  [Tone] Audio-Reasoner loaded")
        except Exception as e:
            print(f"  [Tone] Failed to load Audio-Reasoner: {e}")
            self.engine = None

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行语气识别"""
        if self.engine is None:
            return self._fallback_annotate(audio_path)

        from swift.llm import InferRequest, RequestConfig
        from swift.plugin import InferStats

        # 构建消息
        messages = self._build_messages(audio_path)

        # 推理配置
        request_config = RequestConfig(
            max_tokens=self.config.get('max_tokens', 512),
            temperature=self.config.get('temperature', 0),
            stream=False
        )

        metric = InferStats()

        try:
            # 执行推理
            results = self.engine.infer(
                [InferRequest(messages=messages)],
                request_config,
                metrics=[metric]
            )

            # 解析结果
            if results and len(results) > 0 and results[0] is not None:
                full_response = results[0].choices[0].message.content
                tone_caption = self._extract_caption(full_response)
            else:
                tone_caption = "unknown"

        except Exception as e:
            print(f"  [Tone] Inference failed: {e}")
            tone_caption = "error"

        predictions = {
            "tone_description": tone_caption
        }

        logits_dict = {
            "tone": tone_caption
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _fallback_annotate(self, audio_path: str) -> Dict[str, Any]:
        """降级处理：返回提示信息"""
        predictions = {
            "tone_description": "unavailable",
            "note": "Tone annotation requires Audio-Reasoner environment. Run: conda activate Audio-Reasoner"
        }

        logits_dict = {
            "tone": "unavailable",
            "error": "swift module not installed"
        }

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def _build_messages(self, audio_path: str) -> list:
        """构建推理消息"""
        system_prompt = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": self.TONE_PROMPT}
                ]
            }
        ]

        return messages

    def _extract_caption(self, response: str) -> str:
        """
        提取<CAPTION>内容</CAPTION>
        如果没有CAPTION标签，返回整个响应
        """
        # 尝试提取CAPTION标签内容
        caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
        if caption_match:
            return caption_match.group(1).strip()

        # 尝试提取RESPONSE标签内容
        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
        if response_match:
            return response_match.group(1).strip()

        # 如果都没有，返回清理后的原始响应
        cleaned = re.sub(r'<THINK>.*?</THINK>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip() if cleaned.strip() else response
