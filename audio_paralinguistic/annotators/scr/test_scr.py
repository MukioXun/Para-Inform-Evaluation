#!/usr/bin/env python
"""
SCR (Whisper ASR) 标注器测试脚本
"""
import sys
import os

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))

from audio_paralinguistic.annotators.scr.whisper_asr import WhisperASRAnnotator
from audio_paralinguistic.config.model_config import MODEL_CONFIGS

def test_scr_annotator():
    """测试 SCR 标注器"""
    print("=" * 60)
    print("Testing WhisperASRAnnotator")
    print("=" * 60)

    # 获取配置
    config = MODEL_CONFIGS["SCR"].copy()
    print(f"\n[Config]")
    print(f"  model_name: {config.get('model_name')}")
    print(f"  model_path: {config.get('model_path')}")
    print(f"  language: {config.get('language')}")
    print(f"  task: {config.get('task')}")
    print(f"  device: {config.get('device')}")

    # 创建标注器实例
    annotator = WhisperASRAnnotator(config)

    # 测试1: 检查初始化
    print("\n[Test 1] Initial state")
    print(f"  TASK_NAME: {annotator.TASK_NAME}")
    print(f"  is_loaded: {annotator.is_loaded()}")

    # 测试2: 检查配置参数读取
    print("\n[Test 2] Config parameter reading")
    language = config.get('language', 'zh')
    task = config.get('task', 'transcribe')
    print(f"  language from config: {language}")
    print(f"  task from config: {task}")

    # 测试3: 模型加载 (需要实际模型)
    print("\n[Test 3] Model loading")
    model_path = config.get('model_path')
    if os.path.exists(model_path):
        print(f"  Model path exists: {model_path}")
        try:
            annotator.load_model()
            print(f"  Model loaded successfully: {annotator.is_loaded()}")
        except Exception as e:
            print(f"  Model loading failed: {e}")
    else:
        print(f"  Model path not found: {model_path}")
        print("  Skipping model loading test")

    # 测试4: 查找测试音频
    print("\n[Test 4] Finding test audio")
    test_audio_paths = [
        "/home/u2023112559/qix/datasets/PASM_Lite/000001.mp3",
        "/home/u2023112559/qix/Project/Final_Project/Audio_Captior/audio_paralinguistic/data/input/test.wav"
    ]

    test_audio = None
    for path in test_audio_paths:
        if os.path.exists(path):
            test_audio = path
            print(f"  Found test audio: {path}")
            break

    if test_audio and annotator.is_loaded():
        print("\n[Test 5] Running annotation")
        try:
            result = annotator.annotate(test_audio)
            print(f"  Annotation successful!")
            print(f"  transcription: {result['predictions']['transcription']['text'][:100]}...")
            print(f"  language: {result['predictions']['transcription']['language']}")
        except Exception as e:
            print(f"  Annotation failed: {e}")
    else:
        print("\n[Test 5] Skipping annotation test (model not loaded or no test audio)")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_language_modes():
    """测试不同语言模式配置"""
    print("\n" + "=" * 60)
    print("Testing different language modes")
    print("=" * 60)

    test_configs = [
        {"language": "auto", "task": "transcribe", "description": "Auto-detect language"},
        {"language": "zh", "task": "transcribe", "description": "Chinese transcription"},
        {"language": "en", "task": "transcribe", "description": "English transcription"},
        {"language": "zh", "task": "translate", "description": "Chinese to English translation"},
    ]

    for cfg in test_configs:
        print(f"\n[Config] {cfg['description']}")
        print(f"  language: {cfg['language']}, task: {cfg['task']}")

        # 模拟代码逻辑
        language = cfg['language']
        if language == 'auto':
            mode = "auto-detect mode (no language parameter passed to model)"
        else:
            mode = f"specified language mode (language={language}, task={cfg['task']})"
        print(f"  Expected behavior: {mode}")


if __name__ == "__main__":
    test_language_modes()
    test_scr_annotator()
