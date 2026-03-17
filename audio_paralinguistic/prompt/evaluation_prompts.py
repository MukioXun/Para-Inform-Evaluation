"""
评估相关提示词
"""


class EvaluationPrompts:
    """评估提示词集合"""

    # 聚类解释提示词
    CLUSTER_INTERPRETATION = """请根据以下聚类特征统计，解释这个聚类代表的含义。

聚类特征统计：
{cluster_stats}

请从以下角度进行分析：
1. 这个聚类最显著的特征是什么？
2. 这个聚类可能代表什么样的语音场景？
3. 与其他聚类相比，这个聚类的独特之处？

请用简洁的语言描述（不超过200字）。
"""

    # 类别命名提示词
    CATEGORY_NAMING = """请为以下聚类生成一个简洁的中文名称。

聚类特征：
{features}

要求：
1. 名称不超过8个字
2. 名称要能反映聚类的主要特征
3. 名称要专业且易于理解

只输出名称，不要其他内容。
"""

    # 评估报告生成提示词
    REPORT_GENERATION = """请根据以下分析结果生成一份评估报告。

分析结果：
{analysis_results}

报告要求：
1. 概述整体分析情况
2. 各聚类的主要特点
3. 发现的规律或问题
4. 改进建议

请以专业、清晰的语言撰写报告。
"""

    # 标注质量评估提示词
    ANNOTATION_QUALITY = """请评估以下标注结果的质量。

标注数据：
{annotation_data}

评估维度：
1. 完整性：是否涵盖了所有必要信息
2. 一致性：各项标注是否相互协调
3. 准确性：标注是否合理（基于常识判断）

请给出评分（1-5分）和简要理由。
"""

    # 特征重要性分析提示词
    FEATURE_IMPORTANCE = """请分析以下特征在区分不同类别时的重要性。

特征重要性得分：
{feature_importance}

请解释：
1. 哪些特征最具有区分性？
2. 这些特征如何帮助区分不同类别？
3. 是否存在冗余或无关特征？

请用简洁的语言描述。
"""

    @staticmethod
    def get_prompt(prompt_name: str, **kwargs) -> str:
        """获取并格式化提示词"""
        prompts = {
            "cluster_interpretation": EvaluationPrompts.CLUSTER_INTERPRETATION,
            "category_naming": EvaluationPrompts.CATEGORY_NAMING,
            "report": EvaluationPrompts.REPORT_GENERATION,
            "quality": EvaluationPrompts.ANNOTATION_QUALITY,
            "feature_importance": EvaluationPrompts.FEATURE_IMPORTANCE,
        }

        template = prompts.get(prompt_name, "")
        if template and kwargs:
            return template.format(**kwargs)
        return template
