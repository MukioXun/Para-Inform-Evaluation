#!/usr/bin/env python3
"""
独立的Age+Tone标注脚本
在Audio-Reasoner环境中运行: conda activate Audio-Reasoner

功能:
1. 为已完成的merged结果补充Age标注
2. 为已完成的merged结果补充Tone标注
3. 更新合并结果

使用方法:
    # 在Audio-Reasoner环境中运行
    conda activate Audio-Reasoner
    python run_sar_audio_reasoner.py --input ./data/annotations/merged --audio-dirs /datasets/PASM_Lite

    # 仅标注Age
    python run_sar_audio_reasoner.py --input ./data/annotations/merged --audio-dirs /datasets/PASM_Lite --only age

    # 仅标注Tone
    python run_sar_audio_reasoner.py --input ./data/annotations/merged --audio-dirs /datasets/PASM_Lite --only tone
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# 项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> bool:
    """检查运行环境"""
    try:
        import torch
        print(f"[Info] PyTorch version: {torch.__version__}")
        return True
    except ImportError:
        print("[Error] PyTorch not found!")
        return False


class AgeAnnotator:
    """独立Age标注器"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """加载模型"""
        import torch
        import torch.nn as nn
        import numpy as np
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class ModelHead(nn.Module):
            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = features
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class AgeGenderModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.age = ModelHead(config, 1)
                self.gender = ModelHead(config, 3)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits_age = self.age(hidden_states)
                logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
                return hidden_states, logits_age, logits_gender

        print(f"[Age] Loading model from: {self.model_path}")

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = AgeGenderModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"[Age] Model loaded successfully")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行年龄预测"""
        import torch
        import numpy as np
        import librosa

        if self.model is None:
            return {"error": "Model not loaded"}

        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = audio.astype(np.float32)

        # 预处理
        inputs = self.processor(audio, sampling_rate=16000)
        input_values = inputs['input_values'][0]
        input_values = input_values.reshape(1, -1)
        input_tensor = torch.from_numpy(input_values).to(self.device)

        # 推理
        with torch.no_grad():
            hidden_states, logits_age, logits_gender = self.model(input_tensor)
            normalized_age = logits_age[0, 0].item()

        # 年龄处理
        if 0 <= normalized_age <= 1:
            age_value = normalized_age * 100
        else:
            age_value = normalized_age

        age_group = self._classify_age_group(age_value)

        return {
            "age_value": round(age_value, 2),
            "age_group": age_group,
            "confidence": 0.8 if 5 <= age_value <= 90 else 0.5
        }

    def _classify_age_group(self, age: float) -> str:
        if age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 35:
            return "young_adult"
        elif age < 50:
            return "middle_aged"
        elif age < 65:
            return "senior"
        else:
            return "elderly"


class ToneAnnotator:
    """独立Tone标注器"""

    TONE_PROMPT = """请分析这段语音中说话人的语气特点。

重要提示：
1. 请完全忽略语音中说的具体文字内容
2. 只关注说话人的语气、语调、情感色彩
3. 不要描述语音内容，只描述语气特征

请从以下维度分析语气：
- 情感基调（如：严肃、轻松、激动、平静等）
- 语速特点（如：急促、缓慢、适中、有变化等）
- 语调特征（如：上扬、下沉、平稳、波动等）
- 声音能量（如：有力、柔和、虚弱等）
- 情绪状态（如：自信、犹豫、焦虑、坦然等）

请用简洁的语言总结这段语音的语气特点。"""

    SYSTEM_PROMPT = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.engine = None

    def load_model(self):
        """加载模型"""
        try:
            from swift.llm import PtEngine
            print(f"[Tone] Loading Audio-Reasoner from: {self.model_path}")
            self.engine = PtEngine(
                self.model_path,
                max_batch_size=1,
                model_type='qwen2_audio'
            )
            print(f"[Tone] Model loaded successfully")
        except ImportError:
            print(f"[Tone] Warning: swift module not found!")
            print(f"[Tone] Please run in Audio-Reasoner environment")
            self.engine = None

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行语气标注"""
        if self.engine is None:
            return {"description": "unavailable", "error": "swift not available"}

        from swift.llm import InferRequest, RequestConfig
        from swift.plugin import InferStats

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": self.TONE_PROMPT}
                ]
            }
        ]

        request_config = RequestConfig(max_tokens=512, temperature=0, stream=False)
        metric = InferStats()

        try:
            results = self.engine.infer(
                [InferRequest(messages=messages)],
                request_config,
                metrics=[metric]
            )

            if results and len(results) > 0 and results[0] is not None:
                full_response = results[0].choices[0].message.content
                description = self._extract_caption(full_response)
            else:
                description = "unknown"

        except Exception as e:
            print(f"[Tone] Inference failed: {e}")
            description = "error"

        return {"description": description}

    def _extract_caption(self, response: str) -> str:
        caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
        if caption_match:
            return caption_match.group(1).strip()

        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
        if response_match:
            return response_match.group(1).strip()

        cleaned = re.sub(r'<THINK>.*?</THINK>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip() if cleaned.strip() else response


def find_audio_from_merged(merged_path: Path, audio_dirs: List[Path]) -> Optional[Path]:
    """根据merged文件找到对应的音频文件"""
    audio_id = merged_path.stem.replace('_merged', '')
    extensions = ['.mp3', '.wav', '.flac', '.ogg']

    for audio_dir in audio_dirs:
        for ext in extensions:
            audio_file = audio_dir / f"{audio_id}{ext}"
            if audio_file.exists():
                return audio_file
    return None


def needs_annotation(merged_data: Dict, field: str) -> bool:
    """检查是否需要标注"""
    speaker = merged_data.get('acoustic_features', {}).get('high_level', {}).get('speaker', {})

    if field == 'age':
        age = speaker.get('age', {})
        return not age or not age.get('age_value')

    elif field == 'tone':
        tone = speaker.get('tone', {})
        desc = tone.get('description', '')
        return not tone or desc in ['unavailable', 'unknown', '', None]

    return False


def update_merged(merged_path: Path, age_result: Dict = None, tone_result: Dict = None) -> bool:
    try:
        # 1️⃣ 安全读取
        try:
            with open(merged_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        # 2️⃣ 构造结构
        speaker = (
            data
            .setdefault('acoustic_features', {})
            .setdefault('high_level', {})
            .setdefault('speaker', {})
        )

        # 3️⃣ 更新 Age
        if isinstance(age_result, dict) and 'error' not in age_result:
            speaker['age'] = {
                'age_value': age_result.get('age_value'),
                'age_group': age_result.get('age_group'),
                'confidence': age_result.get('confidence')
            }

        # 4️⃣ 更新 Tone
        if isinstance(tone_result, dict) and 'error' not in tone_result:
            speaker['tone'] = {
                'description': tone_result.get('description')
            }

        # 5️⃣ 原子写入（防损坏）
        tmp_path = merged_path.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        tmp_path.replace(merged_path)

        return True

    except Exception as e:
        print(f"[Error] Failed to update {merged_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Standalone Age+Tone Annotation")

    parser.add_argument("--input", type=str, required=True, help="Merged结果目录")
    parser.add_argument("--audio-dirs", type=str, nargs="+", required=True, help="音频文件目录")
    parser.add_argument("--age-model", type=str,
                        default="/home/u2023112559/qix/Models/Models/age-classification",
                        help="Age模型路径")
    parser.add_argument("--tone-model", type=str,
                        default="/home/u2023112559/qix/Models/Models/Audio-Reasoner",
                        help="Tone模型路径")
    parser.add_argument("--only", choices=["age", "tone"], default=None,
                        help="仅运行指定任务 (默认: 两者都运行)")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    merged_dir = Path(args.input)
    audio_dirs = [Path(d) for d in args.audio_dirs]

    if not merged_dir.exists():
        print(f"[Error] Directory not found: {merged_dir}")
        sys.exit(1)

    # 创建标注器
    run_age = args.only in [None, 'age']
    run_tone = args.only in [None, 'tone']

    age_annotator = None
    tone_annotator = None

    if run_age:
        age_annotator = AgeAnnotator(args.age_model, args.device)
        age_annotator.load_model()

    if run_tone:
        tone_annotator = ToneAnnotator(args.tone_model, args.device)
        tone_annotator.load_model()

    # 处理merged文件
    merged_files = list(merged_dir.glob('*_merged.json'))
    print(f"\n[Info] Found {len(merged_files)} merged files")

    age_count = 0
    tone_count = 0
    skip_count = 0
    fail_count = 0

    for merged_file in merged_files:
        try:
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {merged_file.name}: {e}")
            fail_count += 1
            continue
        audio_path = data.get('file_path')

        if not audio_path:
            print(f"[Warning] 'file_path' missing in: {merged_file.name}")
            fail_count += 1
            continue

        audio_file = Path(audio_path)

        if not audio_file.is_absolute():
            for audio_dir in audio_dirs:
                candidate = audio_dir / audio_file
                if candidate.exists():
                    audio_file = candidate
                    break

        if not audio_file.exists():
            print(f"[Warning] Audio not found: {audio_file}")
            fail_count += 1
            continue

        print(f"[Processing] {audio_file.name}")

        age_result = None
        tone_result = None

        # Age标注
        if run_age and needs_annotation(data, 'age'):
            try:
                age_result = age_annotator.annotate(str(audio_file))
                print(f"  [Age] {age_result.get('age_value')} ({age_result.get('age_group')})")
                age_count += 1
            except Exception as e:
                print(f"  [Age] Failed: {e}")

        # Tone标注
        if run_tone and needs_annotation(data, 'tone'):
            try:
                tone_result = tone_annotator.annotate(str(audio_file))
                print(f"  [Tone] {tone_result.get('description', '')[:50]}...")
                tone_count += 1
            except Exception as e:
                print(f"  [Tone] Failed: {e}")

        # 更新文件
        if age_result or tone_result:
            update_merged(merged_file, age_result, tone_result)
        else:
            skip_count += 1

        if args.limit > 0 and (age_count + tone_count) >= args.limit:
            break

    print(f"\n[Summary] Age: {age_count}, Tone: {tone_count}, Skipped: {skip_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
