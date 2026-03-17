"""
特征配置模块
定义特征维度、融合权重等
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FeatureConfig:
    """特征配置类"""

    # ============ 特征维度定义 ============
    semantic_features: List[str] = field(default_factory=lambda: [
        "text_content",      # 文本内容
        "language",          # 语言
        "intent",            # 意图
        "asr_confidence",    # ASR置信度
    ])

    acoustic_features: List[str] = field(default_factory=lambda: [
        "pitch_mean",        # 音高均值
        "pitch_std",         # 音高标准差
        "energy_mean",       # 能量均值
        "energy_std",        # 能量标准差
        "duration",          # 时长
        "speaking_rate",     # 语速
    ])

    speaker_features: List[str] = field(default_factory=lambda: [
        "speaker_embedding", # 说话人embedding
        "speaker_id",        # 说话人ID
    ])

    emotion_features: List[str] = field(default_factory=lambda: [
        "emotion_category",  # 离散情感类别
        "arousal",           # 唤醒度
        "valence",           # 效价
        "dominance",         # 支配度
        "emotion_intensity", # 情感强度
    ])

    paralingual_features: List[str] = field(default_factory=lambda: [
        "laughter",          # 笑声
        "sigh",              # 叹气
        "cough",             # 咳嗽
        "breath",            # 呼吸声
        "silence_ratio",     # 静音比例
    ])

    # ============ 特征融合权重 ============
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.25,
        "acoustic": 0.20,
        "speaker": 0.15,
        "emotion": 0.30,
        "paralingual": 0.10,
    })

    # ============ 标准化配置 ============
    normalization_method: str = "z-score"  # "z-score" or "min-max"

    # ============ 降维配置 ============
    dimension_reduction: str = "umap"  # "pca", "tsne", "umap"
    n_components: int = 50  # 降维后维度

    def get_all_features(self) -> List[str]:
        """获取所有特征列表"""
        return (
            self.semantic_features +
            self.acoustic_features +
            self.speaker_features +
            self.emotion_features +
            self.paralingual_features
        )

    def get_feature_count(self) -> int:
        """获取特征总数"""
        return len(self.get_all_features())
