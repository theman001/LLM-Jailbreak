"""Streamlit 공용 위젯 — app.py와 pages/가 함께 쓴다."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import streamlit as st

import surgery_engine as se


def data_dir() -> Path:
    """Colab 부트스트랩이 ARD_DATA_DIR(=Google Drive)로 지정. 없으면 로컬 data/."""
    d = Path(os.environ.get("ARD_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_csv() -> Path:
    return data_dir() / "experiment_results.csv"


@st.cache_resource(show_spinner=False)
def get_model():
    with st.spinner(f"{se.MODEL_NAME} 로딩 중… (수 분 소요)"):
        return se.load_model()


def sidebar_params() -> dict:
    """K / Coefficient / Layers. app.py·탭에서 공통 사용."""
    st.sidebar.header("수술 파라미터")
    k = st.sidebar.slider("K (PCA 주성분)", 1, 10, 3)
    coeff = st.sidebar.slider("Coefficient (α, 투영 강도)", 0.0, 2.0, 1.0, 0.05)
    if coeff > se.COEFF_WARN_THRESHOLD:
        st.sidebar.warning(
            f"α > {se.COEFF_WARN_THRESHOLD}: 모델 지능 손상 위험. "
            "탈옥은 되지만 문맥이 무너지면 0.1씩 낮추세요."
        )
    layers = st.sidebar.multiselect("수술 레이어", list(range(0, 32)), default=[14, 16, 18])
    return {"k": k, "coeff": coeff, "layers": sorted(layers)}


@st.fragment(run_every="5s")
def gpu_widget() -> None:
    """메인 추론 루프와 분리된 GPU 모니터 (문서 5 §4)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        used, total = (int(x) for x in out.split(","))
        st.sidebar.metric("GPU VRAM", f"{used/1024:.1f} / {total/1024:.1f} GiB",
                          f"{used/total*100:.0f}%")
    except Exception as e:  # noqa: BLE001 — 모니터가 앱을 죽이면 안 됨
        st.sidebar.caption(f"GPU 상태 조회 불가: {e}")
