import re

def clean_text(text):
    """基础清洗"""
    if not isinstance(text, str):
        return ""
    # 去除 URL
    text = re.sub(r'http\S+', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_segments(text, method='sentence'):
    """
    将长文本切分为片段，用于计算张力弧。
    method: 'sentence' (按句切分) 或 'paragraph' (按段切分)
    """
    if method == 'paragraph':
        segments = text.split('\n')
    else:
        # 简单的按句号切分，保留标点
        segments = re.split(r'(?<=[.!?])\s+', text)
    
    # 过滤掉太短的片段
    return [s for s in segments if len(s.split()) > 3]