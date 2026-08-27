"""ARD — 💬 일반 채팅 (순정 Llama-3, 대조군).

탈옥 실험은 사이드바 → 페이지 "Jailbreak Lab".
"""
import streamlit as st

import surgery_engine as se
from ard_ui import get_model, gpu_widget, sidebar_params

st.set_page_config(page_title="ARD · 일반 채팅", page_icon="💬", layout="wide")
st.title("💬 일반 채팅 (순정 모델)")
st.caption("hook 없는 순정 Llama-3-8B-Instruct. 탈옥 실험의 대조군.")

sidebar_params()  # 페이지 간 파라미터 UI 일관성 유지
gpu_widget()

if st.sidebar.button("모델 로드 / 확인", use_container_width=True):
    get_model()
    st.sidebar.success("모델 준비 완료")

hist = st.session_state.setdefault("chat_standard", [])  # 다른 채팅과 분리된 key
for msg in hist:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    hist.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    model = get_model()
    with st.chat_message("assistant"), st.spinner("생성 중…"):
        reply = se.run_generation(model, se.format_prompt(model, prompt), fwd_hooks=[])
        st.write(reply)
    hist.append({"role": "assistant", "content": reply})
