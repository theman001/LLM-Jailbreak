# CLAUDE.md

## 프로젝트
**Abliteration Research Dashboard (ARD)** — 로컬 오픈웨이트 모델
(`meta-llama/Meta-Llama-3-8B-Instruct`)의 거절(refusal) 서브스페이스를
Rank-k Orthogonal Subspace Projection으로 제거하는 연구를 지원하는
Streamlit 대시보드.

용도: 승인된 모델 해석(interpretability) / 안전성 연구. 로컬 단일 모델 대상.

구현 주체: **Claude Code**. 상세 스펙은 [`document/`](document/) 5개 문서가
단일 진실 원천(single source of truth)이다. 코드를 짜기 전 관련 문서를 읽을 것.

- [1. 프로젝트 개요 및 아키텍처](<document/1. 프로젝트 개요 및 아키텍처 (Project Overview).md>)
- [2. 백엔드 로직 및 수술 스크립트](<document/2. 백엔드 로직 및 수술 스크립트 (Backend Logic).md>)
- [3. UI/UX 요구사항 및 실험 프로세스](<document/3. UI&UX 요구사항 및 실험 프로세스 (UI&UX Spec).md>)
- [4. 대시보드 상세 기능 및 UI 명세서](<document/4. Abliteration 연구 대시보드 상세 기능 및 UI 명세서.md>)
- [5. 알고리즘 및 예외 처리 지침](<document/5. Claude Code를 위한 알고리즘 및 예외 처리 지침.md>)

## 기술 스택
- Python 3.10+ / PyTorch / `transformer_lens` (hooking & intervention)
- `streamlit` (UI), `scikit-learn` (PCA), `numpy`, `pandas`
- **GPU 실행: Google Colab (Pro 권장 — L4 22GB / A100 40GB).** 모델을 로드하는
  모든 작업은 Colab 런타임에서. 무료 T4(16GB)는 OOM 위험. 상세: 문서 5 §8.
- OCI A1은 GPU 없음(ARM CPU) — 코드 편집·경량 작업 전용.
- 연구 CSV는 Google Drive에 저장(Colab 런타임 휘발성).

## 목표 파일 구조 (구현 시)
```
surgery_engine.py      # 거절 서브스페이스 추출 + projection hook (문서 2/4)
app.py                 # Streamlit 엔트리 (일반 채팅 / 탈옥 실험실)
pages/                 # Streamlit multi-page
data/experiment_results.csv   # 연구 로그 (gitignore)
```

## 절대 타협 불가 (Technical Guardrails — 문서 5)
자세한 구현 규칙은 `abliteration-surgery` 스킬에 있다. 요약:

1. **수치 정밀도**: hook 내부 projection 연산은 반드시 `float32`로 업캐스팅 후
   계산하고, 마지막에만 원래 dtype으로 복구. FP16 상태에서 바로 빼지 말 것.
2. **VRAM 세이프가드**: `run_with_cache`는 `names_filter`로 필요한
   `hook_resid_pre`만 캡처하고 사용 직후 `del`. Bulk 실행 시 5문항마다
   `gc.collect()` + `torch.cuda.empty_cache()`. OOM 시 세션 중단 후 캐시 비우기.
3. **데이터 무결성**: `st.data_editor` 점수 수정은 `on_change`로 즉시 CSV에
   원자적 저장. 레코드 ID = `[Timestamp]+[k]+[coeff]+[layers]` 해시.
4. **세션 상태**: 모델은 `st.cache_resource`로 1회 로드. 일반 채팅 / 실시간
   수술 채팅 / 전수 조사 데이터는 서로 다른 key로 분리. 새로고침 시 수동 입력
   점수 유실 금지.
5. **수술 공식**: `h' = h - α · Proj_V(h)`, `V` = PCA k차원 서브스페이스.
   `coeff > 1.2`면 모델 지능 손상 위험 — 사용자에게 경고.

## 개발 규칙
- 새 의존성 추가 전 스택에 이미 있는 것으로 되는지 확인.
- 문서와 코드가 어긋나면 구현을 멈추고 사용자에게 확인.
- 실제 유해 콘텐츠 예시를 코드/커밋/로그에 하드코딩하지 말 것. 데이터셋은
  문서 5의 규격(주제 페어)만 따르고 사용자가 파일로 주입한다.

## 실행
Colab GPU 런타임에서 (Drive 마운트 + HF 토큰 주입 후):
```bash
pip install -r requirements.txt      # 아직 없음 — 첫 구현 시 생성
streamlit run app.py &               # cloudflared/localtunnel로 포트 노출
```
배치 작업(탭 A 스캔 / 탭 C 전수조사)은 Streamlit 없이 노트북 셀로도 실행 가능.
GPU 상태: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader`
