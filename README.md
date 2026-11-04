# 굿홈앤리빙 노동 문의 라우팅 에이전트

직원의 노동 문의를 담당 칸(카테고리)으로 나누고, 그 칸의 근거(가상 사규 + 실제 노동 법령)만 프롬프트에 넣어 답한 뒤, 답변을 코드로 검증하는 [LangGraph](https://langchain-ai.github.io/langgraph/) 에이전트입니다. 데모 화면은 Streamlit으로 만들었습니다.

> **안내**
> - 굿홈앤리빙은 **가상 회사**이며 실존 회사와 무관합니다. `docs/company/`의 사규도 이 프로젝트를 위해 새로 쓴 가상 규정입니다.
> - 이 프로젝트의 답변은 **법률 자문이 아닙니다.** 실제 노동 문제는 고용노동부 고객상담센터(1350)나 전문가에게 확인하세요.

## 동작 흐름

```
질문 → classify(분류) → gate(넘김 판단) ─ 넘김 ─────────────→ handoff → 끝
                                    └ 답변 → retrieve(칸별 근거 조립) → answer(답변)
                                                                         ↓
                                          finalize ← 통과 ─ verify(코드 검증) ─ 실패 → answer 재시도(1회)
                                          (두 번째도 실패하면 확인 불가 안내 문구)
```

1. **분류**: 분류 모델이 주 칸 하나, 보조 칸(없으면 null), 확신도를 냅니다.
2. **넘김 판단**: 코드가 범위 밖이거나 확신도가 `CONF_THRESHOLD`(기본 0.5) 미만이면 담당 창구로 넘깁니다.
3. **칸별 근거 조립**: 고른 칸의 조회 도구(`lookup_*`)만 호출해 그 칸의 사규 조항과 법령 조문만 모읍니다.
4. **답변**: 답변 모델이 받은 근거만으로 답하고 사실마다 근거 id를 붙입니다. 근거에 없는 부분은 따로 돌려줘 그 부분만 넘깁니다.
5. **코드 검증**: 모델 없이 인용한 근거 id와 답에 나온 숫자의 출처를 검사합니다. 실패하면 문제 목록을 주고 한 번 다시 쓰게 합니다.

노드·State·분기 조건을 포함한 자세한 구조도는 [REPORT.md 5절](REPORT.md#5-파이프라인-구조도)에 있습니다.

### 칸 목록

| id | 이름 | 이런 질문 | 조회 도구 |
|---|---|---|---|
| `work` | 근무·휴가·휴직 | 근무시간, 연차, 경조휴가, 출산휴가, 육아휴직 등 | `lookup_work` |
| `pay` | 급여·수당·퇴직금 | 급여일, 연장·야간 수당, 주휴수당, 최저임금, 퇴직금 등 | `lookup_pay` |
| `welfare` | 복리후생 | 식대, 경조금, 직원 할인, 자기계발비, 명절 선물 등 | `lookup_welfare` |
| `contract` | 입사·계약·퇴사 | 근로계약서, 수습, 사직, 해고, 일반 징계 등 | `lookup_contract` |
| `harassment` | 괴롭힘·고충 | 괴롭힘·성희롱 신고, 조사, 피해자 보호 조치 등 | `lookup_harassment` |
| `out_of_scope` | 범위 밖 | 연말정산, 4대보험, 실업급여, 소송 대리 등 | 조회 없이 `handoff` |

칸별로 어떤 조항이 들어가는지는 [docs/mapping.md](docs/mapping.md)와 `data/category_map.json`에 있습니다.

## 폴더 구조

```
.
├── agent.py                  LangGraph 파이프라인(classify → gate → retrieve → answer → verify → finalize)과 CLI
├── context.py                사규·법령을 조 단위로 쪼개고 칸별 근거를 조립하는 조회 도구
├── prompts.py                분류 지침, 답변 규칙, 넘김 문구, 개선 스위치
├── evaluate.py               평가 실행·재채점·비교·채점기 검증
├── app.py                    Streamlit 데모 화면
├── requirements.txt          의존 패키지
├── REPORT.md                 설계·평가셋·측정·구조도 보고서
├── data/
│   ├── category_map.json     칸별로 넣을 사규 조항과 법령 파일
│   ├── examples.json         분류 프롬프트용 예시 10건(평가에 쓰지 않음)
│   ├── goldenset.json        평가셋 50건(프롬프트에 넣지 않음)
│   ├── goldenset_v1.json     처음 만든 정답지(v1, 정리 전 비교용)
│   ├── goldenset_spec.md     평가셋 규격과 정리 원칙
│   ├── judge_validation.json 채점기 검증용 답안 150건
│   ├── _parts/               평가셋 원고(build_goldenset.py 입력)
│   └── _judge_parts/         채점기 검증 답안 원고
├── docs/
│   ├── mapping.md            칸 ↔ 근거 매핑표
│   ├── company/              가상 사규: 취업규칙.md, 복리후생규정.md, 회사소개.md
│   └── law/                  칸별 법령 근거(work/pay/welfare/contract/harassment.md), 색인(index.md)
│       └── raw/              국가법령정보센터에서 받은 법령 원문(txt)
├── scripts/
│   ├── build_goldenset.py    평가셋 병합과 규격 검사
│   ├── cause_counts.py       실행별 실패 원인 건수 집계(API 호출 없음)
│   └── debug_draft.py        문항 하나의 검증 전 초안과 검증 결과 확인
├── tests/
│   ├── test_verify.py        코드 검증기 시험(모델 없이 실행)
│   └── test_evaluate.py      금지 숫자 검사·분류 지표 시험(모델 없이 실행)
└── runs/                     평가 실행 결과(results.jsonl, summary.json, report.md)와 history.csv
```

## 실행 방법

Windows PowerShell 기준입니다. Python 3.13과 [uv](https://docs.astral.sh/uv/)로 확인했습니다.

### 1. 가상환경과 패키지 설치

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

```powershell
$env:OPENAI_API_KEY = "여기에_본인_키"
```

현재 PowerShell 창에서만 유효합니다. 프로젝트 폴더에 `.env` 파일(`OPENAI_API_KEY=...`)을 두어도 읽습니다. `.env`는 `.gitignore`에 들어 있습니다. **API 키는 저장소에 올리지 마세요.**

### 3. 질문 하나 실행

```powershell
python agent.py "결혼하면 휴가 며칠이에요?"
```

분류 결과, 호출한 도구, 검증 결과, 인용한 근거, 최종 답변과 소요 시간·토큰을 출력합니다. 질문을 생략하면 위 예시 질문으로 실행합니다.

### 4. 데모 화면 실행

```powershell
python -m streamlit run app.py
```

질문을 입력하면 답변과 함께 분류·넘김 판단·호출한 도구, 근거로 쓴 조각, 코드 검증 결과를 보여 줍니다. 기본 화면의 **저장된 실행 기록 보기**에서 문항을 선택하면 API 키 없이 답변·분류·도구·인용·넘긴 부분·당시 검증 결과를 재생합니다. **평가 기록 재생**으로 표시하며 모델을 호출하지 않습니다. 근거 탭은 기록의 칸으로 현재 근거를 조립합니다. 새 질문은 별도 모드이며, 예시 질문 버튼은 `data/examples.json`에서 가져옵니다.

### 모델·기준값 바꾸기

환경변수로 바꿉니다. 지정하지 않으면 기본값을 씁니다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `ROUTER_MODEL` | `gpt-5.6-luna` | 분류 모델 |
| `ANSWER_MODEL` | `gpt-5.6-terra` | 답변 모델 |
| `JUDGE_MODEL` | `gpt-5.6-sol` | 평가의 채점 모델(`evaluate.py`에서 사용) |
| `CONF_THRESHOLD` | `0.5` | 분류 확신도가 이 값 미만이면 담당 창구로 넘김 |

```powershell
$env:ANSWER_MODEL = "사용할_모델_이름"
$env:CONF_THRESHOLD = "0.6"
python agent.py "야간근무하면 수당 얼마예요?"
```

## 평가 방법

평가셋 50건(`data/goldenset.json`)으로 세 가지를 잽니다.

| 지표 | 보는 것 |
|---|---|
| 분류 | 주 칸 정확도, macro F1, 혼동 행렬 |
| 도구 호출 적절성 | 실제로 호출한 도구 집합이 기대 집합과 정확히 같은지 |
| 답변 적절성 | 필수 사실을 모두 담고 금지 내용·금지 숫자를 하나도 어기지 않았는지(사실 판단은 채점 모델, 금지 숫자는 코드) |

평가셋 설계와 채점기 검증 과정은 [REPORT.md 3절](REPORT.md#3-평가셋-설계)에 있습니다. 평가는 모델 API를 호출하므로 비용이 듭니다.

```powershell
# 평가셋 전체를 3번 실행 → runs/새실험_r1 ~ r3/ 에 저장, history.csv에 기록
python evaluate.py run --label 새실험 --repeat 3 --note "메모"

# 일부 문항만 실행
python evaluate.py run --label try --only W01,P04

# 저장된 답변을 지금의 정답지로 다시 채점(에이전트는 다시 돌리지 않음)
python evaluate.py regrade --src 05_strict_secondary_r1,05_strict_secondary_r2,05_strict_secondary_r3 --label 05b_gold_v21 --strip 05_strict_secondary

# 여러 실행의 평균·범위 비교
python evaluate.py compare --labels 05_strict_secondary_r1,05_strict_secondary_r2,05_strict_secondary_r3

# 채점기 검증(data/judge_validation.json의 150건) → runs/judge_check.md
python evaluate.py judge-check
```

동시 실행 수는 `EVAL_WORKERS`(기본 6)로 바꿉니다.

### 개선 스위치

**채택: `STRICT_SECONDARY`(데모 기본값).** 데모에서 미지정 시 `1`, 명시적 `0`이면 꺼집니다. CLI·평가에서는 환경변수를 지정하지 않으면 모두 꺼져 있습니다. 나머지 개선 스위치는 꺼 둡니다. `BOUNDARY_EXAMPLES` 실험은 평가 정보 유출로 무효이며, `HANDOFF_STRICT`는 1회 측정에서 필요 없는 넘김을 줄였지만(도구 0.72 → 0.88), 반복 측정 전이라 꺼 둡니다. 값이 `1`일 때만 켜지며, 켜진 스위치는 실행 보고서(`report.md`)에 기록됩니다. 채택 여부는 [REPORT.md 4절](REPORT.md#4-측정-결과)을 참고하세요.

| 환경변수 | 바꾸는 것 |
|---|---|
| `STRICT_SECONDARY=1` | 분류 지침의 보조 칸 기준을 엄격하게 바꿉니다. 서로 다른 칸의 제도 둘을 각각 분명히 물을 때만 보조 칸을 붙입니다. |
| `HANDOFF_STRICT=1` | 답변 출력 형식에서 "근거에 없어 넘길 부분(`uncovered`)"의 정의를 좁힙니다. 근거로 답한 질문의 세부 사항은 넘김으로 치지 않습니다. |
| `ANSWER_RULES_V2=1` | 답변 규칙을 보강합니다. 넘김 기준, 조건·예외·적용 시점 안내, 묻지 않은 제도 덧붙이지 않기, 계산 숫자는 식으로 보이기를 추가합니다. |
| `BOUNDARY_EXAMPLES=1` | 분류 지침에 경계 판정 예(연차 규정 ↔ 퇴사 정산, 경조휴가 ↔ 경조금 등)를 추가합니다. |

```powershell
$env:STRICT_SECONDARY = "1"
python agent.py "결혼하면 휴가 며칠이고 축의금도 나와요?"
Remove-Item Env:STRICT_SECONDARY   # PowerShell 환경변수는 창을 닫기 전까지 남으니 끝나면 지웁니다
```

### 모델 없이 도는 시험

API 키 없이 실행됩니다.

```powershell
python tests/test_verify.py       # 코드 검증기: 근거 id, 숫자 출처, 계산식 판정
python tests/test_evaluate.py     # 금지 숫자 검사, 분류 지표 계산
```

`python tests/test_evaluate.py --judge`는 채점 모델을 실제로 한 번 부르므로 API 키가 필요합니다.

### 보조 스크립트

```powershell
python scripts/build_goldenset.py --check-only   # 평가셋 규격 검사만(--check-only를 빼면 data/_parts를 병합해 goldenset.json 생성)
python scripts/cause_counts.py 05_strict_secondary_r1,05_strict_secondary_r2,05_strict_secondary_r3   # 실행별 실패 원인 집계
python scripts/debug_draft.py F03                # 문항 하나의 검증 전 초안 확인
```

## 측정 결과

정답지 v2·보조 칸 엄격 설정 3회 평균: 분류 정확도 **0.938**, macro F1 **0.942**, 도구 **0.840**(0.80~0.90), 답변 **0.853**(0.84~0.86). 같은 답변을 정답지 v2.1로 재채점하면 답변 **0.847**(0.82~0.88)입니다. 고친 검증기로 다시 돌린 1회는 도구 0.720·답변 0.860, 여기에 `HANDOFF_STRICT`를 켠 1회는 도구 0.880·답변 0.840입니다. 자세한 결과와 한계는 [REPORT.md 4절](REPORT.md#4-측정-결과)을 참고하세요.

## 근거 문서 출처

**사규(가상)**: `docs/company/`의 취업규칙, 복리후생규정, 회사소개는 이 프로젝트를 위해 새로 쓴 가상 문서입니다. 취업규칙은 고용노동부 「2026년 표준 취업규칙」을 뼈대로 하되, 칸 구성에 필요한 장만 골라 회사 고유 규정(숫자 포함)을 넣었습니다.

**법령**: [국가법령정보센터](https://www.law.go.kr)의 현행 조문 원문입니다. 칸별로 필요한 조문만 `docs/law/`에 모았고, 조문 목록은 `docs/law/index.md`에 있습니다.

| 법령 | 판 |
|---|---|
| 근로기준법 | 시행 2026. 10. 8., 법률 제21533호 |
| 근로기준법 시행령 | 시행 2025. 10. 23., 대통령령 제35436호 |
| 근로자퇴직급여 보장법 | 시행 2026. 9. 18., 법률 제21475호 |
| 남녀고용평등과 일ㆍ가정 양립 지원에 관한 법률 | 시행 2026. 9. 18., 법률 제21476호 |
| 최저임금법 | 시행 2020. 5. 26., 법률 제17326호 |
| 2026년 적용 최저임금 고시 | 고용노동부고시 제2025-47호 (시간급 10,320원) |

법령은 개정될 수 있으므로 실제 상담에는 현행 원문을 확인해야 합니다. 근거 문서 선정 과정은 [REPORT.md 1절](REPORT.md#1-주제와-근거-문서)에 있습니다.
