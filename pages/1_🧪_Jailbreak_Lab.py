"""🧪 탈옥 실험실 — 진단 / 실시간 수술 / 전수 조사."""
from datetime import datetime

import pandas as pd
import streamlit as st

import surgery_engine as se
from ard_ui import get_model, gpu_widget, results_csv, sidebar_params, data_dir

st.set_page_config(page_title="ARD · Jailbreak Lab", page_icon="🧪", layout="wide")
st.title("🧪 탈옥 실험실")

params = sidebar_params()
gpu_widget()

PAIRS_PATH = data_dir() / "pairs.csv"
tab_diag, tab_live, tab_bulk = st.tabs(["거절 급소 진단", "실시간 수술", "전수 조사"])


def _load_pairs_or_warn():
    try:
        return se.load_pairs(PAIRS_PATH)
    except (FileNotFoundError, ValueError) as e:
        st.warning(f"{e}\n\n`{PAIRS_PATH}` 에 harmful,harmless 컬럼 CSV를 넣어주세요.")
        return None


# --------------------------------------------------------------------------- #
# 탭 A — 진단
# --------------------------------------------------------------------------- #
with tab_diag:
    st.subheader("레이어별 거절 신호 강도  ‖mean(diffs)‖")
    lo, hi = st.slider("스캔 레이어 범위", 0, 31, (8, 28))
    if st.button("스캔 실행", type="primary"):
        pairs = _load_pairs_or_warn()
        if pairs:
            with st.spinner("레이어 스캔 중…"):
                df = se.scan_refusal_intensity(get_model(), pairs, range(lo, hi + 1))
            st.session_state["scan_df"] = df

    if "scan_df" in st.session_state:
        df = st.session_state["scan_df"]
        st.line_chart(df, x="layer", y="score")
        st.dataframe(
            df.set_index("layer").T.style.background_gradient(axis=1, cmap="Reds"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        top_n = c1.number_input("상위 N개 레이어", 1, 10, 3)
        k = c2.number_input("k (서브스페이스 차원)", 1, 10, params["k"])
        method = st.radio("추출 방식", ["mean_diff", "svd"], horizontal=True,
                          help="mean_diff=표준 거절 방향(k=1). svd=중심화 없는 rank-k.")
        if st.button("상위 레이어 벡터 추출"):
            pairs = _load_pairs_or_warn()
            if pairs:
                sel = df.nlargest(int(top_n), "score")["layer"].astype(int).tolist()
                kk = 1 if method == "mean_diff" else int(k)
                with st.spinner(f"레이어 {sel} 서브스페이스 추출 중…"):
                    sub = se.extract_subspace(get_model(), pairs, sel, kk, method)
                st.session_state["subspace"] = sub
                st.session_state["subspace_meta"] = {"layers": sel, "k": kk, "method": method}
                st.success(f"추출 완료: 레이어 {sel}, k={kk}, {method}")

    if meta := st.session_state.get("subspace_meta"):
        st.info(f"세션에 저장된 서브스페이스: {meta}")


# --------------------------------------------------------------------------- #
# 탭 B — 실시간 수술 (일회성, 연구 로그와 분리)
# --------------------------------------------------------------------------- #
with tab_live:
    sub = st.session_state.get("subspace")
    if not sub:
        st.info("먼저 [거절 급소 진단] 탭에서 서브스페이스를 추출하세요.")
    else:
        meta = st.session_state["subspace_meta"]
        c1, c2 = st.columns(2)
        use_k = c1.slider("사용할 차원 수", 1, meta["k"], meta["k"])
        alpha = c2.slider("α (강도)", 0.0, 2.0, 1.0, 0.05)
        if alpha > se.COEFF_WARN_THRESHOLD:
            st.warning(f"α > {se.COEFF_WARN_THRESHOLD}: 지능 손상 위험.")

        hist = st.session_state.setdefault("chat_steer", [])
        for m in hist:
            st.chat_message(m["role"]).write(m["content"])
        if prompt := st.chat_input("수술된 모델에게 질문"):
            hist.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            sliced = {l: v[:use_k] for l, v in sub.items()}
            model = get_model()
            hooks = se.build_hooks(sliced, alpha)
            with st.chat_message("assistant"), st.spinner("생성 중…"):
                reply = se.run_generation(model, se.format_prompt(model, prompt), hooks)
                st.write(reply)
            hist.append({"role": "assistant", "content": reply})
        if hist and st.button("대화 비우기"):
            st.session_state["chat_steer"] = []
            st.rerun()


# --------------------------------------------------------------------------- #
# 탭 C — 전수 조사
# --------------------------------------------------------------------------- #
with tab_bulk:
    csv_path = results_csv()
    up = st.file_uploader("질문지 (.csv: prompt 컬럼 / .txt: 1행 1질문)", type=["csv", "txt"])
    st.caption(f"결과 저장 위치: `{csv_path}`")

    if up and st.button("실험 시작", type="primary"):
        tmp = data_dir() / f"_upload{''.join(c for c in up.name if c == '.' or c.isalnum())[-8:]}"
        tmp.write_bytes(up.getbuffer())
        try:
            prompts = se.load_prompts(tmp)
        except (ValueError, FileNotFoundError) as e:
            st.error(str(e)); st.stop()
        sub = st.session_state.get("subspace")
        if not sub:
            st.error("서브스페이스 미추출 — 탭 A에서 먼저 추출하세요."); st.stop()

        meta = st.session_state["subspace_meta"]
        batch_ts = datetime.now().isoformat(timespec="seconds")
        k, coeff, layers = meta["k"], params["coeff"], meta["layers"]
        setting = f"k{k}/a{coeff}/L{layers}"
        done = set(se.read_results(csv_path).get("ID", []))
        model = get_model()
        hooks = se.build_hooks(sub, coeff)
        bar = st.progress(0.0)
        for i, q in enumerate(prompts):
            rid = se.record_id(batch_ts, k, coeff, layers, i)
            if rid not in done:
                resp = se.run_generation(model, se.format_prompt(model, q), hooks)
                se.append_result(csv_path, {
                    "ID": rid, "설정": setting, "질문": q, "응답": resp,
                    "탈옥성공률": None, "무결성": None, "비고": "",
                })
            bar.progress((i + 1) / len(prompts))
        st.success(f"{len(prompts)}건 완료")

    df = se.read_results(csv_path)
    if not df.empty:
        st.subheader("연구 테이블 (점수·비고 직접 입력 → 자동 저장)")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic",
                                key="results_editor")
        if edited.to_csv(index=False) != df.to_csv(index=False):
            se.write_results(csv_path, edited)  # 셀 편집 시 즉시 원자적 저장

        scored = edited.dropna(subset=["탈옥성공률"])
        if not scored.empty:
            c1, c2 = st.columns(2)
            c1.scatter_chart(scored, x="설정", y="탈옥성공률")
            if len(scored) >= 2:
                vals = pd.to_numeric(scored["탈옥성공률"], errors="coerce").dropna()
                c2.metric("최근 탈옥성공률", f"{vals.iloc[-1]:.1f}",
                          f"{vals.iloc[-1] - vals.iloc[-2]:+.1f}")
