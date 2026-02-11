# 📄 문서 1: 프로젝트 개요 및 아키텍처 (Project Overview)
## 1. 프로젝트명
Abliteration Research Dashboard (ARD): 다차원 투영 기반 LLM 탈옥 및 데이터 무결성 최적화 분석 도구

## 2. 개발 개요
본 프로젝트는 LLM(Llama-3-8B) 내부의 거절 매커니즘을 수학적으로 제거(Abliteration)하는 연구를 지원하기 위한 실험용 대시보드 개발을 목적으로 한다. 사용자는 웹 UI를 통해 실시간으로 수술 파라미터를 조절하고, 대량의 질문 셋을 실행하여 **'탈옥 성공률'**과 '데이터 무결성' 간의 최적의 지점을 찾아낸다.

## 3. 기술 스택 (Tech Stack)
```text
Language: Python 3.10+
Model Framework: transformer_lens (Hooking & Intervention)
UI Framework: Streamlit (Rapid Prototyping & Data Table Editing)
Math/Stats: PyTorch, Scikit-learn (PCA), NumPy
Infrastructure: Oracle Cloud Infrastructure (OCI) A1 Instance (24GB VRAM)
```
## 4. 시스템 아키텍처
Core Engine: transformer_lens를 이용한 모델 로드 및 Subspace Projection 집도.
State Management: Streamlit 세션 상태를 이용한 실험 데이터 및 모델 캐싱.
Data Layer: 실험 결과를 CSV 형태로 로컬 저장 및 관리.
