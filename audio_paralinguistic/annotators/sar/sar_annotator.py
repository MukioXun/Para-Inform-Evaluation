"""
SAR标注器 - 说话人属性识别
整合: Age Classifier, Gender Classifier, Tone Annotator
"""
import torch
import librosa
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from ..base_annotator import BaseAnnotator


class SARAnnotator(BaseAnnotator):
    """说话人属性标注器 - 整合Age/Gender/Tone"""

    TASK_NAME = "SAR"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sub_annotators = {}
        self.sub_configs = config.get('sub_configs', {})

    def load_model(self):
        """加载所有子模型"""
        print(f"  [SAR] Loading sub-annotators...")

        # 加载Age Classifier
        if self.config.get('enable_age', True):
            from .age_classifier import AgeClassifier
            age_config = self.sub_configs.get('Age', {})
            age_config['device'] = self.device
            self.sub_annotators['Age'] = AgeClassifier(age_config)
            print(f"  [SAR] Loading Age classifier...")
            self.sub_annotators['Age'].load_model()

        # 加载Gender Classifier
        if self.config.get('enable_gender', True):
            from .gender_classifier import GenderClassifier
            gender_config = self.sub_configs.get('Gender', {})
            gender_config['device'] = self.device
            self.sub_annotators['Gender'] = GenderClassifier(gender_config)
            print(f"  [SAR] Loading Gender classifier...")
            self.sub_annotators['Gender'].load_model()

        # 加载Tone Annotator
        if self.config.get('enable_tone', True):
            from .tone_annotator import ToneAnnotator
            tone_config = self.sub_configs.get('Tone', {})
            tone_config['device'] = self.device
            self.sub_annotators['Tone'] = ToneAnnotator(tone_config)
            print(f"  [SAR] Loading Tone annotator...")
            self.sub_annotators['Tone'].load_model()

        print(f"  [SAR] All sub-annotators loaded")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行说话人属性识别"""
        results = {}

        # Age
        if 'Age' in self.sub_annotators:
            try:
                results['Age'] = self.sub_annotators['Age'].annotate(audio_path)
            except Exception as e:
                print(f"  [SAR] Age annotation failed: {e}")
                results['Age'] = {"error": str(e)}

        # Gender
        if 'Gender' in self.sub_annotators:
            try:
                results['Gender'] = self.sub_annotators['Gender'].annotate(audio_path)
            except Exception as e:
                print(f"  [SAR] Gender annotation failed: {e}")
                results['Gender'] = {"error": str(e)}

        # Tone
        if 'Tone' in self.sub_annotators:
            try:
                results['Tone'] = self.sub_annotators['Tone'].annotate(audio_path)
            except Exception as e:
                print(f"  [SAR] Tone annotation failed: {e}")
                results['Tone'] = {"error": str(e)}

        # 整合结果
        return self._merge_results(results)

    def _merge_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """整合各子任务结果"""
        predictions = {
            "attributes": {
                "age": {},
                "gender": {},
                "tone": {}
            }
        }

        logits_dict = {}

        # Age
        if 'Age' in results and 'predictions' in results['Age']:
            age_pred = results['Age']['predictions']
            predictions["attributes"]["age"] = {
                "age_value": age_pred.get("age_value"),
                "age_group": age_pred.get("age_group"),
                "confidence": age_pred.get("confidence")
            }
            logits_dict["age"] = results['Age'].get('logits', {})

        # Gender
        if 'Gender' in results and 'predictions' in results['Gender']:
            gender_pred = results['Gender']['predictions']
            predictions["attributes"]["gender"] = {
                "label": gender_pred.get("gender"),
                "confidence": gender_pred.get("confidence")
            }
            logits_dict["gender"] = results['Gender'].get('logits', {})

        # Tone
        if 'Tone' in results and 'predictions' in results['Tone']:
            tone_pred = results['Tone']['predictions']
            predictions["attributes"]["tone"] = {
                "description": tone_pred.get("tone_description")
            }
            logits_dict["tone"] = results['Tone'].get('logits', {})

        return {
            "predictions": predictions,
            "logits": logits_dict
        }

    def annotate_age(self, audio_path: str) -> Dict[str, Any]:
        """单独执行年龄预测"""
        if 'Age' in self.sub_annotators:
            return self.sub_annotators['Age'].annotate(audio_path)
        return {"error": "Age annotator not loaded"}

    def annotate_gender(self, audio_path: str) -> Dict[str, Any]:
        """单独执行性别预测"""
        if 'Gender' in self.sub_annotators:
            return self.sub_annotators['Gender'].annotate(audio_path)
        return {"error": "Gender annotator not loaded"}

    def annotate_tone(self, audio_path: str) -> Dict[str, Any]:
        """单独执行语气识别"""
        if 'Tone' in self.sub_annotators:
            return self.sub_annotators['Tone'].annotate(audio_path)
        return {"error": "Tone annotator not loaded"}


# 保持向后兼容
SenseVoiceAttributeAnnotator = SARAnnotator
ECAPAAttributeAnnotator = SARAnnotator
