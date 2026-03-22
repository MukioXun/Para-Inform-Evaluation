"""
SAR (Speaker Attribute Recognition) 模块
整合 Age, Gender, Tone 三个子任务
"""
from .sar_annotator import SARAnnotator
from .age_classifier import AgeClassifier
from .gender_classifier import GenderClassifier
from .tone_annotator import ToneAnnotator

# 保持向后兼容
SenseVoiceAttributeAnnotator = SARAnnotator

__all__ = [
    "SARAnnotator",
    "AgeClassifier",
    "GenderClassifier",
    "ToneAnnotator",
    "SenseVoiceAttributeAnnotator"
]
