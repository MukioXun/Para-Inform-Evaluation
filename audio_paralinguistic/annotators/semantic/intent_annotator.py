"""
意图识别标注器
使用 Phi-4-mini 或 Gemma-3n-E2B 进行意图识别
"""
from typing import Dict, Any, List
from ..base_annotator import BaseAnnotator


class IntentAnnotator(BaseAnnotator):
    """
    意图识别标注器

    使用模型: Phi-4-mini / Gemma-3n-E2B
    输出: 意图类别、置信度
    """

    # 预定义意图类别（可根据业务调整）
    DEFAULT_INTENTS = [
        "question",       # 提问
        "statement",      # 陈述
        "command",        # 命令
        "greeting",       # 问候
        "apology",        # 道歉
        "thanks",         # 感谢
        "request",        # 请求
        "agreement",      # 同意
        "disagreement",   # 反对
        "other"           # 其他
    ]

    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-mini-instruct",
        device: str = "cuda",
        intent_list: List[str] = None,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.intent_list = intent_list or self.DEFAULT_INTENTS
        self._feature_names = [
            "intent",
            "intent_confidence",
            "intent_distribution"
        ]

    def load_model(self) -> None:
        """
        加载模型

        依赖: transformers, torch
        """
        # TODO: 实现模型加载
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        raise NotImplementedError(
            "Model loading not implemented. "
            "Please install transformers: pip install transformers torch\n"
            f"Model: {self.model_name}"
        )

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """
        执行意图识别

        需要先进行ASR获取文本，再进行意图分类

        Args:
            audio_path: 音频文件路径

        Returns:
            {
                "intent": "question",
                "intent_confidence": 0.85,
                "intent_distribution": {"question": 0.85, "statement": 0.10, ...}
            }
        """
        if not self.is_loaded():
            self.load_model()

        # TODO: 实现意图识别
        # 需要ASR结果作为输入

        return {
            "intent": "unknown",
            "intent_confidence": 0.0,
            "intent_distribution": {}
        }

    def annotate_from_text(self, text: str) -> Dict[str, Any]:
        """
        从文本进行意图识别

        Args:
            text: ASR转写文本

        Returns:
            意图识别结果
        """
        # TODO: 实现文本意图识别
        pass

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
