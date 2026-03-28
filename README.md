# 🔮 로스트아크 마켓 오라클 (LMO)

> **비정형 패치노트를 딥러닝 언어 모델(LLM)로 정량화하여 MMORPG 아이템 경제 시가 변동 방향성을 예측하는 Neural-Symbolic AI 시스템**

![LMO Dashboard 시연 영상](docs/lmo_demo.webp)

> 💡 **Notice: 대용량 데이터 셋(Raw Data) 미업로드 안내**
> 본 프로젝트의 원천 시계열 데이터 및 LLM 파생 피처셋(`lostark_features.csv`, `고해상도_*.csv` 등)은 **총합 3.5GB 이상의 고빈도(초/분 단위) 대용량 파일**로 구성되어 있습니다. 
> 이는 깃허브 LFS(Large File Storage)의 무료 계정 총합 스토리지 및 대역폭 제한(1GB)을 훌쩍 초과하므로, 채용 당담자분들의 열람 편의성과 레포지토리 경량화를 위해 **원본 데이터 파일은 본 레포지토리 업로드에서 의도적으로 제외**(`.gitignore`)되었습니다.
> 
> 전체 파이프라인의 실시간 연동 및 분석 시뮬레이션 결과물은 소스 코드를 직접 구동하지 않으셔도 되도록 **상단의 대시보드 구동 영상(GIF)**을 통해 확인해 주시기를 바랍니다.

---

## 🚀 1. 핵심 성과 (Key Achievements)
*   **글로벌 방향성 적중률 63.8% 달성**: 무작위성을 철저히 배제한 워크포워드(Walk-Forward) 검증 환경에서 강력한 시장 투사력 입증.
*   **고해상도 데이터 파이프라인**: 98개 핵심 재화 품목 대상 누적 **2,618만 건**의 시계열 가격 데이터 및 **1,500건** 규모의 공식/커뮤니티 이벤트 데이터 수집.
*   **오차율(RMSE) 획기적 감소 (15.2% -> 4.7%)**: 단순(Base) 회귀 모델 대비 하이브리드 LLM 피처 결합과 단조 제약(Monotone Constraints) 보정을 거쳐 다중 시차 예측력 마진 최적화.

---

## 🏗️ 2. 데이터 & AI 아키텍처 (Architecture)
LMO 프로젝트는 전통적인 퀀트(Quant) 시계열 모델이 '비정형 외부 이벤트(공지사항, 유저 반응 등)'를 해석하지 못하는 한계를 극복하고자 다음과 같은 4단계 파이프라인으로 설계되었습니다.

1.  **데이터 수집 (Data Collection)**: 로스트아크 공식 개발자 API (가격 데이터) 및 웹 크롤링 기반 공식 패치노트/인벤 반응 등 실시간 시그널 스크래핑
2.  **LLM 피처 생성 (Enrichment)**: Perplexity API 추론 모델을 통과시켜, 무미건조한 패치노트 문맥을 분석하여 '재련 재료 수요 파급력(0-10점)', '인플레이션 수치' 등 16차원의 수리적 지표로 차원 변환 (Neural-Symbolic 매핑).
3.  **ML 시계열 예측 (Modeling)**: NLP 변환된 지표들과 거시 경제 변수들을 결합하여 XGBoost 엔진에 주입. 최신 메타(5배) 및 고영향 충격량(100배) 가중치와 단조 제약을 설정하여 T+1부터 T+30까지의 입체적 시세 변동 예측.
4.  **인사이트 시각화 (Dashboard)**: Streamlit과 Plotly 기반 대시보드를 통하여 실시간 인게임 투자 추천(AI 매수/매도 시그널) 및 산출 근거를 투명하게 시각적 리포팅.

---

## 📂 3. 디렉터리 및 폴더 구조 (Folder Structure)

본 레포지토리는 확장성에 유리하도록 수집부터 운영까지 모듈형 구조를 채택하고 있습니다.

```text
C:\LMO\
│
├── 📄 dashboard.py           # Streamlit AI 마켓 모니터링 프론트엔드 UI
├── 📄 README.md              # 현재 읽고 계신 프로젝트 대문 설명서
├── 📄 .gitignore             # 가상 환경 및 API Key(보안 파일) 무시 설정
│
├── 📁 data/                  # 수집된 Raw Data 및 LLM이 가공한 Enriched Data 보관
├── 📁 models/                # 훈련이 완료된 XGBoost 인스턴스 피클(.pkl) 파일 보관
│
├── 📁 docs/                  # 기술 및 포트폴리오 문서 모음
│   ├── LMO(Lost Ark Market Oracle) 프로젝트.pptx  # 최종 PPT 포트폴리오
│   ├── ARCHITECTURE.md       # 세부 아키텍처 및 다이어그램 설계도
│   ├── PROJECT_LOG_KR.md     # 전체 프로젝트 진행 및 디버깅 일지 (Dashboard 연동)
│   ├── mechanisms.md         # LLM을 통제하는 게임 경제학적 명제(Knowledge-Base)
│   └── FEATURES.md           # 16차원 정형/비정형 모델 피처 상세 명세서
│
└── 📁 scripts/               # 파이프라인 본체 소스코드 공간
    │
    ├── 📁 collection/        # [자동 수집 모듈]
    │   ├── crawl_inven.py                # 유저 여론 동향 스크래핑
    │   ├── crawl_notices.py              # 로스트아크 공식 패치 크롤러
    │   ├── fetch_official_api.py         # 아이템 시세 추출 엔진
    │   └── trigger_stream_analysis.py    # 전체 수집 자동화 트리거
    │
    ├── 📁 enrichment/        # [AI 차원 정량화 모듈]
    │   └── analyze_events_llm.py         # 패치노트를 16 스코어로 변환하는 LLM 엔진
    │   └── analyze_inven_reaction.py     # 인벤 댓글 NLP 센티멘트 분석기
    │
    ├── 📁 modeling/          # [머신러닝 및 훈련 엔진]
    │   ├── prepare_training_data.py      # 이벤트 및 마켓 데이터 병합(Merge) 로직
    │   └── train_impact.py               # XGBoost 훈련 최적화 (단조 제약, 가중치 적용 등)
    │
    ├── 📁 ops/               # [검증 및 운영 모듈]
    │   └── backtest_model.py             # 시계열 비순환 분할 검증(Data Leakage 차단기)
    │
    └── 📁 utils/             # [유틸리티 및 성능 평가 모듈]
        ├── eval_hit_rate.py              # 글로벌 방향성(63.8%) 계산 평가기
        └── market_validator.py           # 아이템 T3->T4 이상치 필터
```

---

## 🛠️ 4. 기술 스택 (Tech Stack)
*   **언어 / 언어 프레임워크**: Python 3.11
*   **AI & ML 엔진**: XGBoost (MultiOutputRegressor), Scikit-learn
*   **LLM 파싱 엔진**: Perplexity API (Sonar-Reasoning)
*   **데이터 엔지니어링**: Pandas, Joblib, BeautifulSoup4, Lost Ark Developer API
*   **프론트엔드 및 시각화**: Streamlit, Plotly

---

## 💻 5. 로컬 실행 방법 (How to Run Dashboard)
1.  Python 3.11 환경에서 레포지토리를 Clone합니다.
2.  루트 디렉터리(`C:\LMO`)에 `.env` 파일을 생성하고 `PERPLEXITY_API_KEY`와 로스트아크 공식 API 토큰을 발급받아 입력합니다.
3.  터미널에 다음 명령어를 입력하여 대시보드를 구동합니다.
```bash
streamlit run dashboard.py
```
