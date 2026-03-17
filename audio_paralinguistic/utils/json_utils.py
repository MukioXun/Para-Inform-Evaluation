"""
JSON处理工具函数
"""
import json
import re
import os
from typing import Any, Dict, List, Optional


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载JSONL文件

    Args:
        file_path: 文件路径

    Returns:
        字典列表
    """
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
    return items


def save_jsonl(
    items: List[Dict],
    file_path: str,
    mode: str = 'w',
    ensure_ascii: bool = False
) -> None:
    """
    保存为JSONL文件

    Args:
        items: 字典列表
        file_path: 文件路径
        mode: 写入模式
        ensure_ascii: 是否转义非ASCII字符
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, mode, encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=ensure_ascii) + '\n')


def parse_llm_json(raw_string: str) -> Dict:
    """
    解析LLM输出的JSON

    处理markdown代码块等格式

    Args:
        raw_string: 原始字符串

    Returns:
        解析后的字典
    """
    if not raw_string:
        return {}

    # 去除前后空白
    cleaned = raw_string.strip()

    # 处理markdown代码块
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, cleaned)

    if match:
        json_content = match.group(1).strip()
    else:
        json_content = cleaned

    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"待解析内容: {json_content[:200]}...")
        return {}


def merge_jsonl_files(
    input_files: List[str],
    output_file: str,
    key_field: Optional[str] = None
) -> int:
    """
    合并多个JSONL文件

    Args:
        input_files: 输入文件列表
        output_file: 输出文件
        key_field: 用于去重的字段（可选）

    Returns:
        合并后的记录数
    """
    seen_keys = set()
    total_count = 0

    with open(output_file, 'w', encoding='utf-8') as fout:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    try:
                        item = json.loads(line.strip())

                        if key_field and key_field in item:
                            key = item[key_field]
                            if key in seen_keys:
                                continue
                            seen_keys.add(key)

                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                        total_count += 1

                    except json.JSONDecodeError:
                        continue

    return total_count


def filter_jsonl(
    input_file: str,
    output_file: str,
    filter_func
) -> int:
    """
    过滤JSONL文件

    Args:
        input_file: 输入文件
        output_file: 输出文件
        filter_func: 过滤函数 (item) -> bool

    Returns:
        过滤后的记录数
    """
    count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                item = json.loads(line.strip())
                if filter_func(item):
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1
            except json.JSONDecodeError:
                continue

    return count
