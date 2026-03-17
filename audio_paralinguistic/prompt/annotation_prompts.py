"""
标注相关提示词
"""


class AnnotationPrompts:
    """标注提示词集合"""

    # ASR标注提示词
    ASR_TRANSCRIPTION = """请将以下音频转写为文字。
如果音频中包含多个说话人，请标注说话人。
如果存在明显的情感表达，请一并标注。

输出格式（JSON）：
{
    "text": "转写文本",
    "language": "语种代码",
    "speakers": ["说话人1", "说话人2"],
    "emotion": "主要情感"
}
"""

    # 情感识别提示词
    EMOTION_RECOGNITION = """请分析这段音频中的情感特征。

需要识别的内容：
1. 主要情感类别（happy, sad, angry, neutral, fear, surprise, disgust）
2. 情感强度（0-1）
3. 是否存在非口语情感表达（笑声、叹气等）

输出格式（JSON）：
{
    "emotion": "情感类别",
    "intensity": 0.8,
    "non_verbal": "笑声/叹气/无"
}
"""

    # 副语言特征分析提示词
    PARALINGUISTIC_ANALYSIS = """请分析这段音频的副语言特征：

1. 语速：快/正常/慢
2. 音高变化：平稳/有起伏
3. 音质：清晰/沙哑/其他
4. 停顿模式：流畅/频繁停顿
5. 声学环境：安静/嘈杂/室内/室外

输出格式（JSON）：
{
    "speech_rate": "normal",
    "pitch_variation": "moderate",
    "voice_quality": "clear",
    "pause_pattern": "fluent",
    "environment": "indoor"
}
"""

    # 意图识别提示词
    INTENT_CLASSIFICATION = """根据以下文本内容，判断说话者的意图。

可选意图类别：
- question: 提问
- statement: 陈述
- command: 命令
- greeting: 问候
- request: 请求
- agreement: 同意
- disagreement: 反对
- other: 其他

文本：{text}

输出格式（JSON）：
{
    "intent": "意图类别",
    "confidence": 0.95
}
"""

    # 综合分析提示词
    COMPREHENSIVE_ANALYSIS = """请对这段音频进行全面分析，包括以下维度：

【语义维度】
- 转写文本
- 语种
- 意图

【情感维度】
- 情感类别
- 情感强度
- VAD值（效价、唤醒度、支配度）

【副语言维度】
- 语速
- 音高
- 音质
- 停顿

【声学维度】
- 声学环境
- 音频质量

请以结构化JSON格式输出分析结果。
"""

    @staticmethod
    def get_prompt(prompt_name: str) -> str:
        """获取指定名称的提示词"""
        prompts = {
            "asr": AnnotationPrompts.ASR_TRANSCRIPTION,
            "emotion": AnnotationPrompts.EMOTION_RECOGNITION,
            "paralingual": AnnotationPrompts.PARALINGUISTIC_ANALYSIS,
            "intent": AnnotationPrompts.INTENT_CLASSIFICATION,
            "comprehensive": AnnotationPrompts.COMPREHENSIVE_ANALYSIS,
        }
        return prompts.get(prompt_name, "")
