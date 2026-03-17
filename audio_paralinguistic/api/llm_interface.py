"""
统一LLM接口
支持多种LLM后端
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import os


class BaseLLMInterface(ABC):
    """LLM接口基类"""

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """发送聊天请求"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class OpenAIInterface(BaseLLMInterface):
    """OpenAI接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client = None

    def _init_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def chat(self, messages: List[Dict], **kwargs) -> Optional[str]:
        self._init_client()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def is_available(self) -> bool:
        return bool(self.api_key)


class GeminiInterface(BaseLLMInterface):
    """Gemini接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash"
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model
        self.client = None

    def _init_client(self):
        if self.client is None:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)

    def chat(self, messages: List[Dict], **kwargs) -> Optional[str]:
        self._init_client()
        try:
            from google.genai import types

            # 转换消息格式
            contents = []
            for msg in messages:
                contents.append(types.Content(
                    role=msg["role"],
                    parts=[types.Part.from_text(msg["content"])]
                ))

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def is_available(self) -> bool:
        return bool(self.api_key)


class LLMInterface:
    """统一LLM接口"""

    BACKENDS = {
        "qwen": lambda **kwargs: __import__('api.qwen_client', fromlist=['QwenClient']).QwenClient(**kwargs),
        "openai": OpenAIInterface,
        "gemini": GeminiInterface,
    }

    def __init__(
        self,
        backend: str = "qwen",
        **kwargs
    ):
        """
        初始化LLM接口

        Args:
            backend: 后端类型 ("qwen", "openai", "gemini")
            **kwargs: 后端参数
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

        self.backend = backend
        self.client = self.BACKENDS[backend](**kwargs)

    def chat(self, messages: List[Dict], **kwargs) -> Optional[str]:
        """发送聊天请求"""
        return self.client.chat(messages, **kwargs)

    def analyze(self, data: Dict[str, Any], prompt: str) -> Optional[str]:
        """分析数据"""
        full_prompt = prompt.format(data=data)
        return self.chat([{"role": "user", "content": full_prompt}])

    def is_available(self) -> bool:
        """检查是否可用"""
        return self.client.is_available()
