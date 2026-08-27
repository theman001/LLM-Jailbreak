"""ARD 핵심 로직 — 거절 서브스페이스 추출 + projection hook + 생성.

Streamlit 비의존. 구현 규칙은 `abliteration-surgery` 스킬 / document/ 참조.
"""
from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

COEFF_WARN_THRESHOLD = 1.2  # 초과 시 모델 지능 손상 위험 (문서 5 §6)
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
GC_EVERY = 5  # 문서 5 §2


# --------------------------------------------------------------------------- #
# 모델 / 프롬프트
# --------------------------------------------------------------------------- #
def load_model(device: str = "cuda"):
    """HookedTransformer 로드. 호출측이 st.cache_resource로 1회 캐시."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        default_prepend_bos=False,  # 검토 B: 템플릿이 BOS를 포함 → 이중 삽입 방지
        device=device,
    )


def format_prompt(model, user_msg: str) -> str:
    """Llama-3 chat 템플릿 적용 (수동 문자열 금지 — 검토 B)."""
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


# --------------------------------------------------------------------------- #
# 데이터 로드
# --------------------------------------------------------------------------- #
def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    """거절 벡터 추출용 페어. CSV에 harmful, harmless 컬럼 필수 (문서 5 §5)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"페어 파일이 없습니다: {p}. harmful,harmless 컬럼의 UTF-8 CSV가 필요합니다."
        )
    if p.suffix.lower() != ".csv":
        raise ValueError("페어 데이터셋은 harmful,harmless 컬럼의 CSV만 지원합니다.")
    df = pd.read_csv(p)
    missing = {"harmful", "harmless"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")
    pairs = [
        (str(a).strip(), str(b).strip())
        for a, b in zip(df["harmful"], df["harmless"])
        if str(a).strip() and str(b).strip()
    ]
    if not pairs:
        raise ValueError("유효한 페어가 한 건도 없습니다.")
    return pairs


def load_prompts(path: str | Path) -> list[str]:
    """전수 조사(탭 C)용 질문 리스트. CSV(prompt 컬럼) 또는 TXT(1행 1질문)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"질문지 파일이 없습니다: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "prompt" not in df.columns:
            raise ValueError("CSV에 'prompt' 컬럼이 없습니다.")
        prompts = [str(x).strip() for x in df["prompt"]]
    else:
        prompts = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    prompts = [x for x in prompts if x]
    if not prompts:
        raise ValueError("질문이 한 건도 없습니다.")
    return prompts


# --------------------------------------------------------------------------- #
# 활성값 수집 / 서브스페이스
# --------------------------------------------------------------------------- #
def _resid_pre_last_token(model, prompt: str, layers: list[int]) -> dict[int, np.ndarray]:
    names = {f"blocks.{l}.hook_resid_pre" for l in layers}
    with torch.no_grad():
        _, cache = model.run_with_cache(
            prompt, names_filter=lambda n: n in names, prepend_bos=False
        )
    out = {
        l: cache[f"blocks.{l}.hook_resid_pre"][0, -1, :].float().cpu().numpy()
        for l in layers
    }
    del cache
    return out


def _collect_diffs(model, pairs, layers, gc_every: int = GC_EVERY) -> dict[int, np.ndarray]:
    """레이어별 (h_harmful - h_harmless) 마지막 토큰 diff 행렬 [n_pairs, d_model]."""
    acc = {l: [] for l in layers}
    for i, (bad, good) in enumerate(pairs):
        hb = _resid_pre_last_token(model, format_prompt(model, bad), layers)
        hg = _resid_pre_last_token(model, format_prompt(model, good), layers)
        for l in layers:
            acc[l].append(hb[l] - hg[l])
        if (i + 1) % gc_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {l: np.stack(v) for l, v in acc.items()}


def scan_refusal_intensity(model, pairs, layers) -> pd.DataFrame:
    """레이어별 거절 신호 강도 = ‖mean(diffs)‖ (문서 4 탭 A)."""
    diffs = _collect_diffs(model, pairs, list(layers))
    rows = [
        {"layer": l, "score": float(np.linalg.norm(diffs[l].mean(axis=0)))}
        for l in layers
    ]
    return pd.DataFrame(rows)


def _gram_schmidt(mat: np.ndarray) -> np.ndarray:
    """행 벡터들을 정규직교화."""
    out = []
    for v in mat:
        for u in out:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n > 1e-8:
            out.append(v / n)
    return np.stack(out) if out else mat[:0]


def _subspace_from_diffs(diffs: np.ndarray, k: int, method: str) -> np.ndarray:
    """diff 행렬 [n, d] → 정규직교 서브스페이스 [<=k, d].

    method="mean_diff": difference-of-means 방향 1개 (표준, 검토 A).
    method="svd":       중심화 없는 SVD 상위 k개 (sklearn PCA의 중심화 회피).
    """
    if method == "mean_diff":
        v = diffs.mean(axis=0, keepdims=True)
        return v / np.linalg.norm(v)
    if method == "svd":
        _, _, vt = np.linalg.svd(diffs, full_matrices=False)
        return _gram_schmidt(vt[:k])
    raise ValueError(f"알 수 없는 method: {method!r} (mean_diff | svd)")


def extract_subspace(
    model, pairs, layers, k: int = 1, method: str = "mean_diff"
) -> dict[int, torch.Tensor]:
    """레이어별 거절 서브스페이스 {layer: float32 tensor [k, d_model]}."""
    diffs = _collect_diffs(model, pairs, list(layers))
    device = getattr(model.cfg, "device", "cpu")
    return {
        l: torch.tensor(
            _subspace_from_diffs(diffs[l], k, method), dtype=torch.float32, device=device
        )
        for l in layers
    }


# --------------------------------------------------------------------------- #
# Projection hook / 생성
# --------------------------------------------------------------------------- #
def make_projection_hook(vectors: torch.Tensor, coeff: float):
    """h' = h - coeff · Proj_V(h). FP32로 계산 후 원 dtype 복구 (문서 5 §1)."""

    def hook_fn(resid_pre: torch.Tensor, hook=None) -> torch.Tensor:
        orig_dtype = resid_pre.dtype
        h = resid_pre.float()
        v = vectors.to(device=h.device, dtype=torch.float32)
        dots = torch.einsum("...d,kd->...k", h, v)
        h_parallel = torch.einsum("...k,kd->...d", dots, v)
        return (h - coeff * h_parallel).to(orig_dtype)

    return hook_fn


def build_hooks(subspaces: dict[int, torch.Tensor], coeff: float):
    return [
        (f"blocks.{l}.hook_resid_pre", make_projection_hook(v, coeff))
        for l, v in subspaces.items()
    ]


def run_generation(
    model,
    prompt: str,
    fwd_hooks,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    with model.hooks(fwd_hooks=fwd_hooks):
        out = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prepend_bos=False,
            verbose=False,
        )
    return out[len(prompt):] if out.startswith(prompt) else out


# --------------------------------------------------------------------------- #
# 연구 로그 (원자적 저장, 문서 5 §3)
# --------------------------------------------------------------------------- #
def record_id(batch_ts: str, k: int, coeff: float, layers, row_idx: int) -> str:
    key = f"{batch_ts}|{k}|{coeff}|{sorted(int(l) for l in layers)}|{row_idx}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_results(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def write_results(csv_path: str | Path, df: pd.DataFrame) -> None:
    """data_editor on_change 동기화용 전체 덮어쓰기."""
    _atomic_write(df, Path(csv_path))


def append_result(csv_path: str | Path, row: dict) -> None:
    """행 단위 즉시 저장 (전수 조사 체크포인트)."""
    df = read_results(csv_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _atomic_write(df, Path(csv_path))


if __name__ == "__main__":
    # ponytail: 실모델 없이 검증 가능한 수학/IO 경로만 자체 점검
    import tempfile

    # 1) projection hook: FP32 정밀도 + dtype 보존 + coeff=0 항등
    d = 16
    v = torch.zeros(1, d)
    v[0, 0] = 1.0  # 첫 축이 "거절 방향"
    h = torch.randn(2, 3, d, dtype=torch.float16)
    removed = make_projection_hook(v, 1.0)(h)
    assert removed.dtype == torch.float16
    assert removed.float()[..., 0].abs().max() < 1e-2, "평행 성분이 제거되지 않음"
    assert torch.allclose(removed.float()[..., 1:], h.float()[..., 1:], atol=1e-2)
    assert torch.equal(make_projection_hook(v, 0.0)(h), h), "coeff=0은 항등이어야 함"

    # 2) 서브스페이스: mean_diff는 평균 방향, svd는 정규직교 k행
    rng = np.random.default_rng(0)
    base = np.zeros(d)
    base[0] = 3.0
    diffs = base + 0.1 * rng.standard_normal((20, d))
    mv = _subspace_from_diffs(diffs, 1, "mean_diff")
    assert mv.shape == (1, d) and abs(np.linalg.norm(mv) - 1) < 1e-6
    assert mv[0, 0] > 0.9, "mean_diff가 거절 방향을 못 잡음 (중심화 버그 회귀)"
    sv = _subspace_from_diffs(diffs, 3, "svd")
    assert sv.shape == (3, d)
    assert np.allclose(sv @ sv.T, np.eye(3), atol=1e-6), "svd 결과가 정규직교 아님"

    # 3) record_id: 안정성 + 파라미터 민감성
    a = record_id("2026-08-27T00:00", 3, 1.0, [14, 16, 18], 0)
    assert a == record_id("2026-08-27T00:00", 3, 1.0, [18, 14, 16], 0)
    assert a != record_id("2026-08-27T00:00", 3, 1.1, [14, 16, 18], 0)

    # 4) append_result: 항상 valid CSV
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "sub" / "results.csv"
        for i in range(3):
            append_result(fp, {"id": i, "score": i * 2})
        got = pd.read_csv(fp)
        assert list(got["id"]) == [0, 1, 2] and len(got) == 3

    # 5) load_pairs: 컬럼 누락 시 명확한 에러
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "p.csv"
        pd.DataFrame({"harmful": ["x"]}).to_csv(bad, index=False)
        try:
            load_pairs(bad)
            assert False, "harmless 누락인데 통과함"
        except ValueError as e:
            assert "harmless" in str(e)

    print("surgery_engine self-check OK")
