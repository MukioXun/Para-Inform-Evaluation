"""
ASR标注器
使用 SenseVoiceSmall 进行语音识别和初步情感/语种识别
"""
from typing import Dict, Any, List
from ..base_annotator import BaseAnnotator


class ASRAnnotator(BaseAnnotator):
    """
    ASR标注器

    使用模型: SenseVoiceSmall (FunASR)
    输出: 转写文本、情感标签、语种、音频事件
    """

    def __init__(
        self,
        model_name: str = "iic/SenseVoiceSmall",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self._feature_names = [
            "text",           # 转写文本
            "language",       # 语种
            "emotion",        # 情感标签
            "event",          # 音频事件
            "confidence",     # 置信度
        ]

    def load_model(self) -> None:
        """
        加载模型

        依赖: funasr, modelscope
        """
        # TODO: 实现模型加载
        # from funasr import AutoModel
        # self.model = AutoModel(model=self.model_name, device=self.device)
        raise NotImplementedError(
            "Model loading not implemented. "
            "Please install funasr: pip install funasr modelscope\n"
            f"Model: {self.model_name}"
        )

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """
        执行ASR标注

        Args:
            audio_path: 音频文件路径

        Returns:
            {
                "text": "转写文本",
                "language": "zh",
                "emotion": "neutral",
                "event": "speech",
                "confidence": 0.95
            }
        """
        if not self.is_loaded():
            self.load_model()

        # TODO: 实现标注逻辑
        # result = self.model.generate(input=audio_path)
        # return self._parse_result(result)

        return {
            "text": "",
            "language": "unknown",
            "emotion": "unknown",
            "event": "unknown",
            "confidence": 0.0
        }

    def _parse_result(self, raw_result) -> Dict[str, Any]:
        """解析模型输出"""
        # TODO: 根据实际模型输出格式解析
        pass

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
