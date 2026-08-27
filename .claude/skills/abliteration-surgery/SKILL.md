---
name: abliteration-surgery
description: >-
  transformer_lens hook으로 Llama-3의 거절 서브스페이스를 제거하는
  surgery_engine 구현/수정 시의 기술 가드레일. Rank-k orthogonal subspace
  projection, PCA 거절 벡터 추출, FP32 정밀도 규칙, VRAM 세이프가드,
  Streamlit 세션/CSV 무결성. surgery_engine.py, projection hook, 거절 벡터
  스캔, 탈옥 실험실 탭, bulk research 로직을 다룰 때 사용.
---

# Abliteration Surgery Engine — 구현 가드레일

전체 스펙: [`document/`](../../../document/) 문서 2·4·5. 이 스킬은 실수하기
쉬운 부분만 압축한다.

## 1. 거절 서브스페이스 추출

```python
def get_refusal_subspace(model, harmful_prompts, harmless_prompts, layer_idx, k=3):
    diffs = []
    for p_bad, p_good in zip(harmful_prompts, harmless_prompts):
        with torch.no_grad():
            name = f"blocks.{layer_idx}.hook_resid_pre"
            _, cache_bad  = model.run_with_cache(p_bad,  names_filter=lambda n: n.endswith(name))
            _, cache_good = model.run_with_cache(p_good, names_filter=lambda n: n.endswith(name))
            d = cache_bad[name][0, -1, :].float() - cache_good[name][0, -1, :].float()
            diffs.append(d.cpu().numpy())
            del cache_bad, cache_good          # 즉시 해제
    pca = PCA(n_components=k)
    pca.fit(np.stack(diffs))
    return torch.tensor(pca.components_, dtype=torch.float32, device="cuda")
```

- 마지막 토큰 위치(`[0, -1, :]`)의 활성값 차이만 사용.
- 프롬프트는 Llama-3 chat 템플릿을 적용한 상태여야 함 (문서 2의 `apply_template`).
- 대조군은 "주제는 유사 / 의도는 안전"한 페어 (문서 5 §5). 유해 문자열을
  소스에 하드코딩하지 말고 사용자가 업로드한 파일에서 읽는다.

## 2. Projection Hook — FP32 순서 엄수

```python
def make_robust_projection_hook(vectors, coeff=1.0):   # vectors: [k, d_model] fp32
    def hook_fn(resid_pre, hook):
        h = resid_pre.float()                                   # 1) FP32 업캐스팅
        dots = torch.einsum("bsd,kd->bsk", h, vectors)
        h_parallel = torch.einsum("bsk,kd->bsd", dots, vectors) # 2) 서브스페이스 투영
        return (h - coeff * h_parallel).to(resid_pre.dtype)     # 3) 마지막에만 복구
    return hook_fn
```

- **하지 말 것**: `resid_pre - coeff * proj` 를 FP16 상태에서 계산. 반복 시
  수치 오차가 누적되어 모델 전체 지능이 붕괴한다.
- PCA 성분이 이미 정규직교라면 `h_parallel` 이 곧 정사영. 직접 만든 벡터를
  넣는 경로가 생기면 Gram–Schmidt 로 정규직교화 후 사용.
- `coeff`(α) > 1.2 → 지능 손상 경고를 UI에 노출. 횡설수설 시 0.1 단위로 하향.

## 3. VRAM 세이프가드 (Colab GPU, ~22–40GB)

실행 환경은 Google Colab GPU 런타임 (문서 5 §8). 공유 GPU라 세이프가드가 더 중요.

- 모델 로드는 앱 전체에서 1회 (`st.cache_resource`).
- Bulk research: 매 5문항 생성 후 `gc.collect()` + `torch.cuda.empty_cache()`.
- 생성 루프 각 반복 끝에서 중간 텐서/캐시 `del`.
- OOM 발생 시: 현재 세션 중단 → `empty_cache()` → 사용자에게 `target_layers`
  개수 또는 `k` 하향 안내.
- 거절 강도 스캔(탭 A)은 Layer 8~28 범위, 점수 = `‖mean(diffs)‖`.

## 4. Streamlit 상태 / 데이터 무결성

- 세 컨텍스트(일반 채팅 · 실시간 수술 채팅 · 전수 조사)는 **다른**
  `st.session_state` key. 혼선 금지.
- 추출된 서브스페이스(레이어별 PCA 성분)는 세션에 저장해 재추출 방지.
- 연구 테이블 점수: `st.data_editor` 의 `on_change` 콜백에서 즉시 CSV 에
  atomic write (임시파일 → `os.replace`). 새로고침/세션 끊김에도 유실 금지.
- CSV 는 **Google Drive 경로**에 둔다 (Colab 런타임 휘발성). 전수 조사는 행
  단위 체크포인트 — 재실행 시 이미 있는 `ID` 는 건너뛴다.
- 레코드 ID = hash(`[Timestamp] + [k] + [coeff] + [layer_list]`).
- 업로드 규격(문서 5 §5): `.csv` 는 `prompt` 컬럼 필수, `.txt` 는 1행 1질문.
  파싱 에러를 최우선 방어 — 잘못된 파일에 앱이 죽지 않게 한다.

## 5. GPU 모니터링

```
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader
```
메인 추론 루프와 분리된 주기적 리프레시로 표시.
