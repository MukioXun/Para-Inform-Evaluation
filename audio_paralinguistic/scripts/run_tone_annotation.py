#!/usr/bin/env python3
"""
独立的Tone标注脚本
在Audio-Reasoner环境中运行: conda activate Audio-Reasoner

功能:
1. 扫描已完成的merged结果文件
2. 为缺少tone标注的文件补充tone
3. 更新合并结果

使用方法:
    # 在Audio-Reasoner环境中运行
    conda activate Audio-Reasoner
    python run_tone_annotation.py --input ./data/annotations/merged

    # 指定音频目录直接标注
    python run_tone_annotation.py --audio-dir /path/to/audio --output ./tone_results
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


def check_swift_available() -> bool:
    """检查swift模块是否可用"""
    try:
        from swift.llm import PtEngine
        return True
    except ImportError:
        print("[Error] swift module not found!")
        print("[Error] Please run in Audio-Reasoner environment:")
        print("[Error]   conda activate Audio-Reasoner")
        return False


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
        from swift.llm import PtEngine

        print(f"[Tone] Loading Audio-Reasoner from: {self.model_path}")
        self.engine = PtEngine(
            self.model_path,
            max_batch_size=1,
            model_type='qwen2_audio'
        )
        print(f"[Tone] Model loaded successfully")

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行语气标注"""
        if self.engine is None:
            return {"error": "Model not loaded"}

        from swift.llm import InferRequest, RequestConfig
        from swift.plugin import InferStats

        # 构建消息
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

        # 推理配置
        request_config = RequestConfig(
            max_tokens=512,
            temperature=0,
            stream=False
        )
        metric = InferStats()

        try:
            # 执行推理
            results = self.engine.infer(
                [InferRequest(messages=messages)],
                request_config,
                metrics=[metric]
            )

            # 解析结果
            if results and len(results) > 0 and results[0] is not None:
                full_response = results[0].choices[0].message.content
                tone_caption = self._extract_caption(full_response)
            else:
                tone_caption = "unknown"

        except Exception as e:
            print(f"[Tone] Inference failed: {e}")
            tone_caption = "error"

        return {
            "description": tone_caption,
            "raw_response": full_response if 'full_response' in dir() else None
        }

    def _extract_caption(self, response: str) -> str:
        """提取<CAPTION>内容"""
        # 尝试提取CAPTION标签内容
        caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
        if caption_match:
            return caption_match.group(1).strip()

        # 尝试提取RESPONSE标签内容
        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
        if response_match:
            return response_match.group(1).strip()

        # 清理并返回
        cleaned = re.sub(r'<THINK>.*?</THINK>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip() if cleaned.strip() else response


def find_audio_from_merged(merged_path: Path, audio_dirs: List[Path]) -> Optional[Path]:
    """根据merged文件找到对应的音频文件"""
    audio_id = merged_path.stem.replace('_merged', '')

    # 支持的音频格式
    extensions = ['.mp3', '.wav', '.flac', '.ogg']

    # 在各个音频目录中搜索
    for audio_dir in audio_dirs:
        for ext in extensions:
            audio_file = audio_dir / f"{audio_id}{ext}"
            if audio_file.exists():
                return audio_file

    return None


def needs_tone_annotation(merged_data: Dict) -> bool:
    """检查是否需要tone标注"""
    speaker = merged_data.get('acoustic_features', {}).get('high_level', {}).get('speaker', {})
    tone = speaker.get('tone', {})

    # 如果tone为空或只有description字段但值为空/unavailable
    if not tone:
        return True

    description = tone.get('description', '')
    if not description or description in ['unavailable', 'unknown', '']:
        return True

    return False


def update_merged_with_tone(merged_path: Path, tone_result: Dict) -> bool:
    """更新merged文件，添加tone标注"""
    try:
        with open(merged_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保路径存在
        if 'acoustic_features' not in data:
            data['acoustic_features'] = {}
        if 'high_level' not in data['acoustic_features']:
            data['acoustic_features']['high_level'] = {}
        if 'speaker' not in data['acoustic_features']['high_level']:
            data['acoustic_features']['high_level']['speaker'] = {}

        # 更新tone
        data['acoustic_features']['high_level']['speaker']['tone'] = {
            'description': tone_result.get('description', 'unknown')
        }

        # 保存
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f"[Error] Failed to update {merged_path}: {e}")
        return False


def process_merged_dir(
    annotator: ToneAnnotator,
    merged_dir: Path,
    audio_dirs: List[Path],
    limit: int = 0
):
    """处理merged目录中的文件"""
    # 查找所有merged文件
    merged_files = list(merged_dir.glob('*_merged.json'))
    print(f"\n[Tone] Found {len(merged_files)} merged files")

    processed = 0
    skipped = 0
    failed = 0

    for merged_file in merged_files:
        # 读取数据检查是否需要标注
        try:
            with open(merged_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {merged_file}: {e}")
            failed += 1
            continue

        # 检查是否需要tone标注
        if not needs_tone_annotation(data):
            skipped += 1
            continue

        # 找到对应的音频文件
        audio_file = find_audio_from_merged(merged_file, audio_dirs)
        if audio_file is None:
            print(f"[Warning] Audio not found for: {merged_file.name}")
            failed += 1
            continue

        print(f"[Tone] Processing: {audio_file.name}")

        # 执行标注
        try:
            tone_result = annotator.annotate(str(audio_file))
            print(f"[Tone]   Result: {tone_result.get('description', 'N/A')[:50]}...")

            # 更新merged文件
            if update_merged_with_tone(merged_file, tone_result):
                processed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"[Error] Annotation failed: {e}")
            failed += 1

        # 限制处理数量
        if limit > 0 and processed >= limit:
            break

    print(f"\n[Tone] Summary: processed={processed}, skipped={skipped}, failed={failed}")


def process_audio_dir(
    annotator: ToneAnnotator,
    audio_dir: Path,
    output_dir: Path
):
    """直接处理音频目录"""
    # 查找音频文件
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
        audio_files.extend(audio_dir.glob(ext))

    print(f"\n[Tone] Found {len(audio_files)} audio files")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        # 执行标注
        try:
            tone_result = annotator.annotate(str(audio_file))

            # 保存结果
            result_data = {
                "audio_id": audio_file.stem,
                "tone": tone_result
            }

            output_path = output_dir / f"{audio_file.stem}_tone.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            print(f"[Tone]   Saved: {output_path}")

        except Exception as e:
            print(f"[Error] Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Tone Annotation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode",
        choices=["merged", "audio"],
        default="merged",
        help="运行模式: merged=处理merged目录, audio=直接处理音频"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="输入目录 (merged模式: merged目录; audio模式: 音频目录)"
    )

    parser.add_argument(
        "--audio-dirs",
        type=str,
        nargs="+",
        help="音频文件目录 (merged模式需要，用于查找音频)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./tone_results",
        help="输出目录 (audio模式)"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/u2023112559/qix/Models/Models/Audio-Reasoner",
        help="Audio-Reasoner模型路径"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制处理数量 (0=不限制)"
    )

    args = parser.parse_args()

    # 检查swift
    if not check_swift_available():
        sys.exit(1)

    # 创建标注器
    annotator = ToneAnnotator(args.model_path)
    annotator.load_model()

    if args.mode == "merged":
        # 处理merged目录
        if not args.input:
            print("[Error] --input required for merged mode")
            sys.exit(1)

        merged_dir = Path(args.input)

        # 音频目录
        audio_dirs = []
        if args.audio_dirs:
            audio_dirs = [Path(d) for d in args.audio_dirs]
        else:
            # 尝试从项目结构推断
            default_audio_dirs = [
                PROJECT_ROOT / "data" / "input",
                Path("/datasets/PASM_Lite"),
                Path("/home/u2023112559/qix/datasets/PASM_Lite"),
            ]
            audio_dirs = [d for d in default_audio_dirs if d.exists()]

            if not audio_dirs:
                print("[Warning] No audio directories found, please specify --audio-dirs")

        process_merged_dir(annotator, merged_dir, audio_dirs, args.limit)

    elif args.mode == "audio":
        # 直接处理音频
        if not args.input:
            print("[Error] --input required for audio mode")
            sys.exit(1)

        audio_dir = Path(args.input)
        output_dir = Path(args.output)

        process_audio_dir(annotator, audio_dir, output_dir)

    print("\n[Tone] Done!")


if __name__ == "__main__":
    main()
