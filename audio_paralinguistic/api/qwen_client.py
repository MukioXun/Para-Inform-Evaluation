"""
千帆大模型客户端
封装阿里云千帆大模型API
"""
import os
import time
import json
import requests
from typing import Dict, Any, Optional, List


class QwenClient:
    """千帆大模型客户端"""

    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    DEFAULT_MODEL = "qwen-turbo"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retry: int = 3,
        retry_sleep: float = 0.5,
        timeout: int = 60
    ):
        """
        初始化千帆客户端

        Args:
            api_key: API密钥（默认从环境变量获取）
            endpoint: API端点
            model: 模型名称
            max_retry: 最大重试次数
            retry_sleep: 重试间隔
            timeout: 请求超时
        """
        self.api_key = api_key or os.getenv("QWEN_API_KEY", "")
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        self.model = model
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Optional[str]:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            模型回复文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        for attempt in range(self.max_retry):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']

            except Exception as e:
                print(f"API请求失败 (尝试 {attempt + 1}/{self.max_retry}): {e}")
                if attempt < self.max_retry - 1:
                    time.sleep(self.retry_sleep)

        return None

    def analyze_annotation(
        self,
        annotation_data: Dict[str, Any],
        prompt_template: Optional[str] = None
    ) -> Optional[str]:
        """
        使用大模型分析标注结果

        Args:
            annotation_data: 标注数据
            prompt_template: 提示词模板

        Returns:
            分析结果
        """
        if prompt_template is None:
            prompt_template = """请分析以下音频标注数据，总结其特点：

{data}

请从以下几个维度进行分析：
1. 情感特征
2. 语言特点
3. 副语言特征
4. 其他显著特点
"""

        prompt = prompt_template.format(data=json.dumps(annotation_data, ensure_ascii=False, indent=2))
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)

    def induce_category_name(
        self,
        cluster_features: Dict[str, Any]
    ) -> Optional[str]:
        """
        推导类别名称

        Args:
            cluster_features: 聚类特征统计

        Returns:
            类别名称
        """
        prompt = f"""根据以下聚类特征统计，为这个类别生成一个简洁的中文名称（不超过10个字）：

特征统计：
{json.dumps(cluster_features, ensure_ascii=False, indent=2)}

只需要输出类别名称，不要其他内容。
"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=0.3, max_tokens=50)

    def generate_description(
        self,
        category_name: str,
        cluster_features: Dict[str, Any]
    ) -> Optional[str]:
        """
        生成类别描述

        Args:
            category_name: 类别名称
            cluster_features: 聚类特征统计

        Returns:
            类别描述
        """
        prompt = f"""类别名称：{category_name}

特征统计：
{json.dumps(cluster_features, ensure_ascii=False, indent=2)}

请为这个类别生成一段简短的描述（不超过100字），说明其主要特征。
"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=0.5, max_tokens=200)

    def batch_analyze(
        self,
        items: List[Dict[str, Any]],
        prompt_template: str
    ) -> List[Optional[str]]:
        """
        批量分析

        Args:
            items: 待分析项列表
            prompt_template: 提示词模板

        Returns:
            分析结果列表
        """
        results = []
        for item in items:
            result = self.analyze_annotation(item, prompt_template)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """检查API是否可用"""
        return bool(self.api_key)
