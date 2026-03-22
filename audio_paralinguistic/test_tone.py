#!/usr/bin/env python
"""
Tone Annotator 测试脚本
需要在 Audio-Reasoner 环境中运行:
    conda activate Audio-Reasoner
    python test_tone.py
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tone():
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
    from swift.plugin import InferStats
    import re

    # 模型配置
    model_path = '/home/u2023112559/qix/Models/Models/Audio-Reasoner'
    audio_path = '/home/u2023112559/qix/datasets/PASM_Lite/000001.mp3'

    # Tone prompt
    tone_prompt = """请分析这段语音中说话人的语气特点。

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

    system_prompt = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'

    # 构建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": tone_prompt}
            ]
        }
    ]

    print(f"Loading Audio-Reasoner from: {model_path}")

    # 创建引擎
    engine = PtEngine(model_path, max_batch_size=1, model_type='qwen2_audio')

    print(f"Processing: {audio_path}")

    # 推理配置
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=False)
    metric = InferStats()

    # 执行推理
    results = engine.infer([InferRequest(messages=messages)], request_config, metrics=[metric])

    if results and len(results) > 0 and results[0] is not None:
        full_response = results[0].choices[0].message.content
        print(f"\n=== Full Response ===\n{full_response}\n")

        # 提取CAPTION
        caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', full_response, re.DOTALL)
        if caption_match:
            caption = caption_match.group(1).strip()
            print(f"=== Extracted Tone Caption ===\n{caption}\n")
        else:
            print("No CAPTION tag found in response")
    else:
        print("No response generated")

if __name__ == '__main__':
    test_tone()
