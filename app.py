import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re

# 页面基础配置
st.set_page_config(
    page_title="Sensing Suspicion", 
    page_icon="🕵️‍♀️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 核心工具函数 ---

@st.cache_resource
def load_model():
    # --- 修改前 (本地路径) ---
    # model_path = "models/creepy_roberta"
    
    # --- 修改后 (Hugging Face ID) ---
    # 请把下面的 "你的HF用户名/模型名" 换成你刚才创建的真实 ID
    model_path = "KiaraLi2025/creepy-roberta" 
    
    st.write(f"Loading model from Hugging Face: {model_path} ...") # 加个提示让我们知道它在工作
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading failed!\nError message: {e}")
        return None, None

def get_prediction_score(text, tokenizer, model):
    """返回 'Creepy' (Label 1) 的概率 (0.0 - 1.0)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()

def split_into_segments(text):
    """简单的分句逻辑"""
    # 按 . ! ? 以及换行符切分
    segments = re.split(r'(?<=[.!?\n])\s+', text)
    return [s.strip() for s in segments if len(s) > 5] # 过滤掉太短的

# --- 界面逻辑 ---

# 加载模型
tokenizer, model = load_model()

st.title("🕵️‍♀️ Sensing Suspicion")
st.markdown("### A Neural Network for Detecting 'Creepy Signals'")

with st.sidebar:
    st.write("### About This Project")
    st.info(
        "This project aims to use **RoBERTa** to detect subtle *creepy signals* in everyday narratives. "
        "By training the model on Reddit posts from subreddits such as **r/LetsNotMeet** and "
        "**r/TwoSentenceHorror**, it learns linguistic patterns associated with unsafe or suspicious "
        "situations, allowing us to predict whether new text may indicate a potentially dangerous encounter."
    )
    st.write("---")
    st.write("**Accuracy:** 95.2%")
    st.write("**Status:** Trained & Ready")

# 主界面布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Narrative")
    user_input = st.text_area(
        "Paste a story here to analyze...", 
        height=400, 
        placeholder="It was late at night, and I was walking home alone..."
    )
    
    analyze_btn = st.button("🔍 Analyze for Suspicion", type="primary")

if analyze_btn and user_input and tokenizer and model:
    # 1. 全局分析
    global_score = get_prediction_score(user_input, tokenizer, model)
    
    # 2. 逐句分析 (用于张力弧)
    segments = split_into_segments(user_input)
    segment_scores = []
    
    # 进度条
    progress_bar = st.progress(0)
    for i, seg in enumerate(segments):
        score = get_prediction_score(seg, tokenizer, model)
        segment_scores.append(score)
        progress_bar.progress((i + 1) / len(segments))
    progress_bar.empty()

    # --- 右侧结果展示 ---
    with col2:
        st.subheader("📊 Analysis Results")
        
        # 仪表盘
        score_color = "red" if global_score > 0.7 else "orange" if global_score > 0.4 else "green"
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color:{score_color}; font-size: 60px; margin:0;">{global_score:.1%}</h1>
                <p>Creepy Index (Overall)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        if global_score > 0.7:
            st.error("⚠️ **High Suspicion Detected!** The model flagged significant unsafe patterns.")
        elif global_score > 0.4:
            st.warning("🤔 **Unsettling Tone.** Some parts of the story feel suspicious.")
        else:
            st.success("✅ **Safe Narrative.** This reads like a normal everyday story.")

        st.divider()

        # 张力弧可视化
        st.markdown("#### 📈 Narrative Tension Arc")
        chart_data = pd.DataFrame({
            'Segment': range(1, len(segments) + 1),
            'Creepy Score': segment_scores,
            'Text': [s[:50] + "..." for s in segments] # 缩略文本
        })
        
        fig = px.line(chart_data, x='Segment', y='Creepy Score', 
                      markers=True, 
                      hover_data=['Text'],
                      line_shape='spline') # 平滑曲线
        
        # 红色警戒区
        fig.add_hrect(y0=0.8, y1=1.0, line_width=0, fillcolor="red", opacity=0.1)
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    # --- 底部：高亮文本展示 ---
    st.divider()
    st.subheader("🔦 Contextual Highlighter")
    st.caption("Sentences flagged as 'Creepy' (>70%) are highlighted in red.")

    annotated_text = ""
    for seg, score in zip(segments, segment_scores):
        if score > 0.7:
            # 红色高亮
            annotated_text += f'<span style="background-color: #ffcccc; padding: 2px 5px; border-radius: 5px; border: 1px solid #ff0000;">{seg}</span> '
        elif score > 0.4:
             # 黄色高亮
            annotated_text += f'<span style="background-color: #fff4cc; padding: 2px 5px; border-radius: 5px;">{seg}</span> '
        else:
            annotated_text += f'{seg} '
            
    st.markdown(f'<div style="line-height: 1.6;">{annotated_text}</div>', unsafe_allow_html=True)

elif analyze_btn and not user_input:
    st.warning("Please paste some text first!")