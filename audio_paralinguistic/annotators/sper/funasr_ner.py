"""
SpER标注器 - FunASR ASR
语音实体识别，使用ASR转录后提取文本信息
"""
import os
import re
import torch
import librosa
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..base_annotator import BaseAnnotator


class FunASRNERAnnotator(BaseAnnotator):
    """FunASR ASR + 简化NER标注器"""

    TASK_NAME = "SpER"

    def load_model(self):
        """加载FunASR ASR模型"""
        from funasr import AutoModel

        print(f"  [SpER] Loading FunASR ASR model...")

        # 使用Paraformer语音识别模型
        self.model = AutoModel(
            model="paraformer-zh",
            device=self.device,
            disable_update=True,
            disable_log=True
        )

        # 简单的实体识别规则（正则表达式）
        self.entity_patterns = {
            "TIME": r'\d{1,2}[点时分秒]|\d{4}年|\d{1,2}月|\d{1,2}日|今天|明天|昨天|上午|下午|晚上|早上',
            "MONEY": r'\d+元|\d+块|\d+万|\d+亿|人民币|美元|欧元',
            "DATE": r'\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日',
            "QUANTITY": r'\d+个|\d+只|\d+条|\d+次|\d+件',
            "LOC": r'北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆|天津|苏州|无锡|东莞|佛山|厦门|福州|济南|青岛|郑州|长沙|合肥|南昌|昆明|贵阳|南宁|海口|三亚|沈阳|哈尔滨|长春|石家庄|太原|呼和浩特|兰州|银川|西宁|乌鲁木齐|拉萨',
        }

        print(f"  [SpER] FunASR ASR loaded")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行语音识别和实体识别"""
        # FunASR ASR推理
        result = self.model.generate(input=audio_path)

        # 解析结果
        entities = []
        entity_summary = {
            "PER": [],
            "LOC": [],
            "ORG": [],
            "TIME": [],
            "MONEY": [],
            "PRODUCT": [],
            "DATE": [],
            "QUANTITY": []
        }

        raw_text = ""
        if result and len(result) > 0:
            res = result[0]
            raw_text = res.get('text', '')

            # 使用正则表达式提取实体
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.finditer(pattern, raw_text)
                for match in matches:
                    entity_text = match.group()
                    entities.append({
                        "entity_text": entity_text,
                        "entity_type": entity_type,
                        "start_char": match.start(),
                        "end_char": match.end(),
                        "confidence": 0.8
                    })
                    if entity_type in entity_summary:
                        entity_summary[entity_type].append(entity_text)

        predictions = {
            "entities": entities,
            "entity_summary": entity_summary,
            "raw_text": raw_text
        }

        logits = {
            "entity_count": len(entities),
            "raw_text_length": len(raw_text)
        }

        return {
            "predictions": predictions,
            "logits": logits
        }


# 别名
SpERAnnotator = FunASRNERAnnotator
