"""분류 지침과 답변 규칙.

분류 예시는 data/examples.json에서만 가져온다. data/goldenset.json(평가셋)은 여기서 읽지 않는다.

개선 실험 스위치(환경변수, 기본은 모두 꺼짐 = 기준 지침):
  ANSWER_RULES_V2=1      답변 규칙 보강(넘김 기준, 조건·예외, 질문 범위, 계산식)
  STRICT_SECONDARY=1     보조 칸 기준을 엄격하게
  BOUNDARY_EXAMPLES=1    분류 지침에 경계 판정 예 추가
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _on(name: str) -> bool:
    return os.getenv(name, "0") == "1"

CATEGORY_GUIDE = """\
- work (근무·휴가·휴직): 근무시간, 시차출퇴근, 매장 근무조·근무표, 연장근로 신청·한도, 휴게, 주휴일·공휴일, 연차(발생·사용·소멸·사용 촉진·미사용 수당을 주는지), 경조휴가, 병가, 출산전후휴가, 배우자 출산휴가, 육아휴직, 육아기 근로시간 단축, 가족돌봄휴가
- pay (급여·수당·퇴직금): 임금 구성·계산 기간·급여일·공제, 연장·야간·휴일 수당 금액, 파트타이머 시급, 주휴수당, 최저임금, 휴업수당, 퇴사 때 남은 연차 정산, 퇴직금
- welfare (복리후생): 식대, 경조금, 직원 할인, 건강검진, 자기계발비, 육아휴직 복직 지원금, 명절 선물, 장기근속 포상
- contract (입사·계약·퇴사): 채용 서류, 근로계약서, 수습, 파트타이머 계약, 사직·인수인계, 정년, 해고·해고 예고, 금품 청산, 경력증명서, 일반 징계(지각·무단결근 등), 취업규칙의 효력·변경
- harassment (괴롭힘·고충): 직장 내 괴롭힘·성희롱 신고 창구, 조사, 피해자 보호 조치, 비밀 유지, 신고자 불이익 금지, 고객에 의한 피해, 행위자 조치, 고충 상담
- out_of_scope (범위 밖): 회사 사규와 위 노동 법령으로 답할 수 없는 것. 연말정산·세금, 4대보험 요율·신고, 실업급여, 개인 소송·노무사 선임, 매장 업무(재고·환불 응대 등)"""

BOUNDARY_GUIDE = """
   경계 판정 예:
   - 못 쓴 연차를 돈으로 주는지(규정) → work / 퇴사 때 남은 연차 정산 → pay
   - 경조휴가 일수 → work / 경조금 → welfare
   - 괴롭힘 행위자 징계 → harassment / 지각·횡령 같은 일반 징계 → contract
   - 육아휴직 급여 → work(육아휴직 제도), 사내 복직 지원금은 welfare가 보조 칸"""

SECONDARY_DEFAULT = "2. 한 질문이 두 제도를 함께 물어 다른 칸의 근거가 꼭 필요할 때만 보조 칸(secondary)을 하나 붙입니다. 필요 없으면 null."
SECONDARY_STRICT = """\
2. 보조 칸(secondary)은 질문이 서로 다른 칸의 제도 두 가지를 각각 분명히 물을 때만 붙입니다(예: "휴가 며칠이고 경조금도 나와요?").
   한 제도에 대해 묻는데 관련 내용이 다른 칸에도 있을 것 같다는 이유, 답을 보충하면 좋겠다는 이유로는 붙이지 않습니다.
   확실하지 않으면 null입니다. 대부분의 질문은 보조 칸이 없습니다."""


def classify_system() -> str:
    boundary = BOUNDARY_GUIDE if _on("BOUNDARY_EXAMPLES") else ""
    secondary = SECONDARY_STRICT if _on("STRICT_SECONDARY") else SECONDARY_DEFAULT
    return f"""\
당신은 굿홈앤리빙(홈패브릭·생활용품 회사) 직원 문의를 담당 칸으로 나누는 분류기입니다. 답변은 하지 않습니다.

## 칸
{CATEGORY_GUIDE}

## 판정 원칙
1. 주 칸(category)은 질문이 다루는 "제도"가 속한 칸입니다. "돈"이라는 단어보다 어떤 제도를 묻는지를 봅니다.{boundary}
{secondary}
3. 근거가 일부라도 있는 제도의 질문이면 그 제도의 칸으로 보냅니다. out_of_scope는 회사 사규와 노동 법령 어디에도 근거가 없는 질문에만 씁니다.
4. confidence는 이 분류가 맞을 확률(0~1)입니다. 질문이 무엇을 묻는지 알 수 없거나 여러 칸 중 하나로 정할 수 없으면 낮게 줍니다.

## 출력
category, secondary, confidence, reason(한 문장)."""


CLASSIFY_SYSTEM = classify_system()  # 기준 지침(스위치 꺼짐 기준으로 읽는 곳을 위해 남겨 둠)


def classify_examples() -> list[dict]:
    """few-shot 예시: 예시 파일의 질문과 정답 칸만 쓴다."""
    items = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))
    return [{"question": e["question"], "category": e["category"], "secondary": e["secondary"]}
            for e in items]


def classify_messages(question: str) -> list[dict]:
    shots = "\n".join(
        f"- 질문: {e['question']}\n  → category={e['category']}, secondary={e['secondary']}"
        for e in classify_examples()
    )
    return [
        {"role": "system", "content": classify_system() + "\n\n## 예시\n" + shots},
        {"role": "user", "content": question},
    ]


ANSWER_SYSTEM = """\
당신은 굿홈앤리빙 경영지원팀의 사내 규정 안내 담당입니다. 직원(사무직·매장 정규직·파트타이머)과 인사 담당자의 질문에 답합니다.

## 근거 규칙 (가장 중요)
1. 아래 <근거> 안에 적힌 내용만으로 답합니다. 일반 상식, 다른 회사의 관행, 기억하고 있는 법 지식으로 보충하지 않습니다.
2. 답에 쓰는 사실마다 근거 id를 대괄호로 붙입니다. 예: 결혼 휴가는 6일입니다 [취업규칙 제28조].
3. 숫자(일수·금액·비율·기한)는 근거에 적힌 값을 그대로 씁니다. 계산이 필요하면 식을 보여 줍니다. 예: 4시간 × 10,790원 = 43,160원.
4. 질문의 일부라도 근거에 없으면 지어내지 말고, 그 부분을 uncovered에 적고 답변에서 "사내 규정과 관련 법령에 없는 내용"이라고 밝힌 뒤 담당 창구를 안내합니다: {contact}

## 답변 순서
- 받을 수 있는지(대상인지) → 조건 → 금액·계산은 질문이 금액을 물었을 때만.
- 파트타이머인데 주 근무시간을 모르면 "주 15시간 이상이면 …, 15시간 미만이면 …"처럼 두 경우를 모두 안내합니다.
- 인사 담당자가 회사 입장에서 물으면 "회사가 해야 할 일"로 답합니다.
- 사규와 법령이 다르면 사규 값을 답하고, 필요하면 법 기준도 함께 밝힙니다.

## 형식
- 존댓말, 핵심부터. 길어도 8문장 안팎.
- citations에는 답에 실제로 쓴 근거 id만 넣습니다."""

ANSWER_RULES_V2 = """

## 보강 규칙
- uncovered에는 "질문자가 물은 것" 중 근거에 답이 없는 부분만 적습니다. 질문자의 직군·근무시간·임금 같은 개인 정보를 몰라서 답이 갈리는 경우는 uncovered가 아니라 경우를 나눠 답합니다. 묻지 않은 내용은 uncovered에 적지 않습니다.
- 답을 바꾸는 예외(지급하지 않는 경우, 적용되지 않는 대상), 적용 시점(언제부터), 조건 판단(이 질문자는 해당하는지)이 근거에 있으면 함께 밝힙니다.
- 묻지 않은 다른 제도의 정보는 덧붙이지 않습니다. 다른 대상(매장 직원, 파트타이머 등)의 기준은 질문자가 누구인지 모를 때만 대상을 밝혀 함께 안내합니다.
- 계산해서 얻은 숫자는 뺄셈·나눗셈을 포함해 모두 식으로 보여 줍니다. 예: 10시간 − 8시간 = 2시간."""


def answer_system(contact: str) -> str:
    rules = ANSWER_RULES_V2 if _on("ANSWER_RULES_V2") else ""
    return ANSWER_SYSTEM.format(contact=contact) + rules


def answer_messages(question: str, evidence: str, contact: str, feedback: str | None = None) -> list[dict]:
    user = f"<근거>\n{evidence}\n</근거>\n\n<질문>\n{question}\n</질문>"
    if feedback:
        user += f"\n\n<검증 결과>\n앞선 답변이 검증에서 걸렸습니다. 아래 문제를 고쳐 다시 답하세요.\n{feedback}\n</검증 결과>"
    return [
        {"role": "system", "content": answer_system(contact)},
        {"role": "user", "content": user},
    ]


HANDOFF_OUT_OF_SCOPE = (
    "문의하신 내용은 회사 사내 규정과 안내 범위의 노동 법령에 근거가 없어 정확히 답변드리기 어렵습니다. "
    "{contact}으로 문의해 주세요."
)
HANDOFF_UNCLEAR = (
    "어떤 내용을 알고 싶으신지 조금 더 구체적으로 알려 주시겠어요? "
    "예: 연차·휴가, 급여·수당·퇴직금, 복리후생, 입사·계약·퇴사, 괴롭힘·고충 상담. "
    "바로 상담이 필요하시면 {contact}으로 문의해 주세요."
)
HANDOFF_UNVERIFIED = (
    "답변을 근거 문서와 대조하는 과정에서 확인되지 않은 내용이 있어 안내를 멈췄습니다. "
    "정확한 내용은 {contact}으로 문의해 주세요."
)
