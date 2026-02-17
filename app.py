import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title='영화 리뷰 감성 분석', page_icon="🧐")

@st.cache_resource
def load_model():
    model_path = "./final_model" 
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
 
    return model, tokenizer

with st.spinner("AI 모델을 불러오는 중입니다..."):
    model, tokenizer = load_model()

st.title("🧐 Positive or Negartive?")
st.markdown("### 영화 리뷰 감성 분석기")
st.write("문장을 입력하시면 **KcELECTRA**기반 모델이 문장의 긍정/부정을 판단합니다.")

# 입력창
user_input = st.text_area("분석할 문장을 입력해보세요:", height=100, placeholder="예: 영화가 재미있어요~")


if st.button("Assess the review"):
    if user_input.strip() == "":
        st.warning("문장을 입력해주세요!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=64) 
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        neg_prob = probs[0][0].item()
        pos_prob = probs[0][1].item()
        
        st.divider() 
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="긍정 확률(👍)", value=f"{pos_prob*100:.1f}%")
        with col2:
            st.metric(label="부정 확률(❌)", value=f"{neg_prob*100:.1f}%")

        # 최종 판단
        if pos_prob > 0.5:
            st.success(f"**결과: 긍정적인 리뷰입니다!** 👍")
        else:
            st.error(f"**결과: 부정적인 리뷰입니다.** ❌")
