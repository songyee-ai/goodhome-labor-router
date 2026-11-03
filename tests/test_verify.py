"""기계적 검증(check_grounding)과 verify 노드 시험. 모델을 부르지 않는다.

실행: .venv\\Scripts\\python.exe tests/test_verify.py
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402
import context  # noqa: E402
from agent import check_grounding  # noqa: E402

EVIDENCE = ('<근거 id="취업규칙 제28조">본인 결혼 6일</근거> 시급 10,790원, 1일 8시간, 1주 40시간, '
            '한도 470,000원, 1,000원 미만은 버린다, 통상임금 100분의 50, 적용 기간: 2026. 1. 1.~2026. 12. 31.')
IDS = ["취업규칙 제28조"]
QUESTION = "주 18시간 일해요. 주휴일에 10시간 근무했고 통상시급은 20,000원이에요"

CASES = [
    # (이름, 답변, citations, 통과해야 하나)
    ("근거에 있는 값", "결혼 휴가는 6일입니다 [취업규칙 제28조].", [], True),
    ("지어낸 값", "결혼 휴가는 5일입니다 [취업규칙 제28조].", [], False),
    ("맞는 계산식(이어진 식)", "18 ÷ 40 × 8 = 3.6시간이고, 3.6시간 × 10,790원 = 38,844원입니다.", [], True),
    ("틀린 계산식", "18 ÷ 40 × 8 = 3.6시간이고, 3.6시간 × 10,790원 = 38,000원입니다.", [], False),
    ("지어낸 재료로 만든 식", "5일 × 2 = 10일입니다.", [], False),
    ("버림 계산", "470,000원 × 18 ÷ 40 = 211,000원(1,000원 미만 버림)입니다.", [], True),
    ("받지 않은 근거 인용", "6일입니다 [근로기준법 제60조].", ["근로기준법 제60조"], False),
    ("질문에 있는 숫자", "주 18시간이면 대상입니다.", [], True),
    ("넘김 연락처", "고용노동부 고객상담센터 1350으로 문의하세요.", [], True),
    # 아래는 기준 측정에서 실제 답변이 쓴 모양(괄호, 이어진 등호, 백분율, 굵게, 마침표 붙은 날짜, 세미콜론 인용)
    ("괄호 식", "올해 한도는 470,000원 × (18시간 ÷ 40시간) = 211,500원이며, 1,000원 미만 금액은 버립니다.", [], True),
    ("괄호·이어진 등호·백분율",
     "8시간을 넘는 시간은 10시간 − 8시간 = 2시간입니다. "
     "(8시간 × 20,000원 × 50%) + (2시간 × 20,000원 × 100%) = 80,000원 + 40,000원 = **120,000원**입니다.", [], True),
    ("이어진 식의 합계를 틀림",
     "8시간을 넘는 시간은 10시간 − 8시간 = 2시간입니다. "
     "(8시간 × 20,000원 × 50%) + (2시간 × 20,000원 × 100%) = 80,000원 + 40,000원 = 130,000원입니다.", [], False),
    ("식 없이 머릿속으로 뺀 값", "8시간 초과 2시간분에는 100%를 가산합니다.", [], False),
    ("영문 x 곱셈", "10시간 - 8시간 = 2시간이고, 2시간 x 20,000원 x 1.0 = 40,000원입니다.", [], True),
    ("마침표 붙은 날짜", "적용 기간은 2026년 1월 1일부터 12월 31일까지입니다.", [], True),
    ("세미콜론 인용", "결혼 휴가는 6일입니다 [취업규칙 제28조; 취업규칙 제28조].", [], True),
    ("버림 말이 멀리 있으면 버림 불인정",
     "3.6시간 × 10,790원 = 38,000원입니다. 15시간 미만 파트타이머는 대상이 아닙니다.", [], False),
]

# 실제 근거(context.assemble + render)로 하는 시험. (이름, 칸, 질문, 답변, citations, 통과해야 하나)
Q0 = "궁금해요"
REAL_CASES = [
    # 결함 1: 근거의 조 번호·날짜가 허용 목록에 들어가 지어낸 값이 통과하던 문제
    ("work 맞음: 결혼 휴가 6일", "work", Q0, "본인 결혼 휴가는 6일입니다 [취업규칙 제28조].", [], True),
    ("work 맞음: 자녀 결혼 2일, 120일 이내", "work", Q0,
     "자녀 결혼은 2일이고, 결혼 휴가는 예식일과 혼인신고일 중 빠른 날부터 120일 이내에 나누어 쓸 수 있습니다 [취업규칙 제28조].",
     [], True),
    ("work 맞음: 가족돌봄 처음 2일 유급", "work", Q0, "가족돌봄휴가 중 매년 처음 2일은 유급입니다 [취업규칙 제37조].", [], True),
    ("work 맞음: 병가 3일 이상 진단서", "work", Q0, "3일 이상 연속하여 병가를 쓰면 진단서를 냅니다.", [], True),
    ("work 지어냄: 결혼 휴가 7일", "work", Q0, "본인 결혼 휴가는 7일입니다 [취업규칙 제28조].", [], False),
    ("work 지어냄: 37일(제37조의 37)", "work", Q0, "가족돌봄휴가는 37일입니다 [취업규칙 제37조].", [], False),
    ("contract 맞음: 수습 2개월", "contract", Q0, "정규직 수습 기간은 입사일부터 2개월입니다 [취업규칙 제8조].", [], True),
    ("contract 맞음: 서류 7일 이내", "contract", Q0, "입사일부터 7일 이내에 서류를 제출합니다 [취업규칙 제6조].", [], True),
    ("contract 맞음: 징계위원회 5명", "contract", Q0,
     "징계위원회는 3명 + 2명 = 5명으로 구성합니다 [취업규칙 제74조].", [], True),
    ("contract 지어냄: 수습 9일", "contract", Q0, "수습 기간은 9일입니다 [취업규칙 제8조].", [], False),
    ("contract 지어냄: 수습 4개월", "contract", Q0, "수습 기간은 4개월입니다 [취업규칙 제8조].", [], False),
    ("welfare 맞음: 식대 187,000원", "welfare", Q0, "정규직 식대는 월 187,000원입니다 [복리후생규정 제4조].", [], True),
    ("welfare 맞음: 할인 32%·한도 1,600,000원", "welfare", Q0,
     "자사 상품을 32% 할인받고, 정규직 연간 한도는 1,600,000원입니다 [복리후생규정 제6조].", [], True),
    ("welfare 맞음: 경조금 60일 이내", "welfare", Q0, "경조금은 60일 이내에 신청합니다 [복리후생규정 제5조].", [], True),
    ("welfare 맞음: 상반기 6월 30일까지", "welfare", Q0, "상반기분은 6월 30일까지 신청합니다 [복리후생규정 제8조].", [], True),
    ("welfare 지어냄: 식대 180,000원", "welfare", Q0, "정규직 식대는 월 180,000원입니다 [복리후생규정 제4조].", [], False),
    ("welfare 지어냄: 6월 31일", "welfare", Q0, "상반기분은 6월 31일까지 신청합니다 [복리후생규정 제8조].", [], False),
    ("pay 맞음: 야간 60%", "pay", Q0, "야간근로는 통상임금의 60%를 가산합니다 [취업규칙 제46조].", [], True),
    ("pay 맞음: 퇴직금 10일 이내", "pay", Q0, "퇴직금은 퇴직일부터 10일 이내에 지급합니다 [취업규칙 제59조].", [], True),
    ("pay 맞음: 오후 10시~오전 6시", "pay", Q0, "오후 10시부터 다음 날 오전 6시까지가 야간근로입니다 [취업규칙 제46조].", [], True),
    ("pay 맞음: 최저임금 절 이름 인용", "pay", Q0, "2026년 시간급 최저임금은 10,320원입니다 [2026년 적용 최저임금].", [], True),
    ("pay 맞음: 적용 기간 날짜", "pay", Q0, "적용 기간은 2026년 1월 1일부터 12월 31일까지입니다 [2026년 적용 최저임금].", [], True),
    ("pay 지어냄: 퇴직금 12일", "pay", Q0, "퇴직금은 퇴직일부터 12일 이내에 지급합니다 [취업규칙 제59조].", [], False),
    ("pay 지어냄: 야간 65%", "pay", Q0, "야간근로는 통상임금의 65%를 가산합니다 [취업규칙 제46조].", [], False),
    ("pay 지어냄: 적용 시작 3월 1일", "pay", Q0, "2026년 3월 1일부터 적용합니다 [2026년 적용 최저임금].", [], False),
    # 결함 2: 계산식이 스스로를 근거로 만들던 문제
    ("식 재료 지어냄: 1234 + 0", "work", Q0, "1234 + 0 = 1234일입니다.", [], False),
    ("식 재료 지어냄: 37일 × 1", "work", Q0, "37일 × 1 = 37일입니다.", [], False),
    ("결과에서 거꾸로 푼 식: 2일 + 8일 = 10일", "work", Q0,
     "처음 2일은 유급이고 나머지 8일은 무급입니다(2일 + 8일 = 10일) [남녀고용평등법 제22조의2].", [], True),
    ("재료 둘을 모름: 4.5일 + 5.5일 = 10일", "work", Q0, "4.5일 + 5.5일 = 10일 중 4.5일은 유급입니다.", [], False),
    # 결함 3: 100분의 N의 다른 표현
    ("pay 맞음: 150%", "pay", Q0, "연장근로는 통상임금의 150%를 받습니다 [취업규칙 제45조].", [], True),
    ("pay 맞음: 1.5배", "pay", Q0, "연장근로는 통상시급의 1.5배입니다 [취업규칙 제45조].", [], True),
    ("pay 맞음: 200%", "pay", Q0, "휴일 8시간을 넘는 시간은 200%입니다 [취업규칙 제47조].", [], True),
    ("pay 맞음: × 1.6", "pay", "통상시급이 15,000원이에요",
     "야간 시급은 15,000원 × 1.6 = 24,000원입니다 [취업규칙 제46조].", [], True),
    ("pay 지어냄: 175%", "pay", Q0, "연장근로는 통상임금의 175%입니다 [취업규칙 제45조].", [], False),
    # 결함 4: 한국어 금액 표기
    ("welfare 맞음: 13만 원", "welfare", Q0, "정규직 명절 선물은 13만 원 상당입니다 [복리후생규정 제10조].", [], True),
    ("welfare 맞음: 18만 7천 원", "welfare", Q0, "정규직 식대는 월 18만 7천 원입니다 [복리후생규정 제4조].", [], True),
    ("welfare 맞음: 47만 원 × 18 ÷ 40 (버림)", "welfare", "주 18시간 파트타이머예요",
     "연간 한도는 47만 원 × 18 ÷ 40 = 211,500원이고 1,000원 미만은 버려 211,000원입니다 [복리후생규정 제8조].",
     [], True),
    ("welfare 지어냄: 99만 9천 원", "welfare", Q0, "지원금은 99만 9천 원입니다 [복리후생규정 제9조].", [], False),
    ("welfare 지어냄: 8만 원", "welfare", Q0, "파트타이머 명절 선물은 8만 원 상당입니다 [복리후생규정 제10조].", [], False),
    # 결함 5: 식 안의 낱말, 마침표로 붙은 두 식
    ("pay 맞음: 주휴수당 계산", "pay", "주 15시간 일하는 파트타이머예요",
     "주휴수당은 (15시간 ÷ 40시간) × 8시간 × 10,790원 = 32,370원입니다 [취업규칙 제52조].", [], True),
    ("pay 맞음: 식 안의 낱말(주·하루)", "pay", "주 18시간 일하고 하루 7시간씩 주 3일 근무해요",
     "(주 18시간 ÷ 40시간) × 8시간 = 3.6시간이고, 하루 7시간 × 주 3일 = 21시간입니다 [취업규칙 제52조].", [], True),
    ("pay 맞음: 마침표로 붙은 두 식", "pay", "주휴일에 10시간 일했고 통상시급은 23,456원이에요",
     "10시간 − 8시간 = 2시간. 2시간 × 23,456원 × 100% = 46,912원입니다 [취업규칙 제47조].", [], True),
    ("pay 맞음: 등호 없이 쓴 식", "pay", "주 2일, 하루 7시간 일해요",
     "주 2일 × 하루 7시간이면 1주 소정근로시간은 14시간이므로 15시간 미만입니다 [취업규칙 제4조].", [], True),
    ("pay 맞음: 마감조 20분 야간수당", "pay", "22시 20분에 끝나요. 10시 넘은 20분도 수당이 붙나요? 통상시급이 15,000원이에요",
     "오후 10시부터 10시 20분까지 20분 ÷ 60분 × 15,000원 × 60% = 3,000원입니다 [취업규칙 제46조].", [], True),
    ("work 맞음: 10시 출근 19시 퇴근", "work", "10시에 출근하면 몇 시 퇴근이에요?",
     "출근 시각부터 9시간 뒤인 19:00(오후 7시)에 퇴근합니다 [취업규칙 제13조].", [], True),
    ("pay 틀린 식: 마침표 뒤 식의 결과를 틀림", "pay", "주휴일에 10시간 일했고 통상시급은 23,456원이에요",
     "10시간 − 8시간 = 2시간. 2시간 × 23,456원 × 100% = 46,000원입니다 [취업규칙 제47조].", [], False),
    # 남은 한계: 같은 칸의 다른 조문에 같은 값+단위가 있으면 단위 대조만으로는 막지 못한다
    ("한계: 수습 3일(제62조 3근무일과 같은 값)", "contract", Q0, "수습 기간은 3일입니다 [취업규칙 제8조].", [], True),
    ("한계: 퇴직금 14일(퇴직급여법 제9조의 값)", "pay", Q0, "퇴직금은 14일 이내에 지급합니다 [취업규칙 제59조].", [], True),
    # 결함 6: eval 거듭제곱
    ("거듭제곱(××) 불인정", "pay", Q0, "2××10 = 1024원입니다.", [], False),
    # 결함 7: 흔한 인용 모양
    ("인용: 조·항", "work", Q0, "본인 결혼 휴가는 6일입니다 [취업규칙 제28조 제1항].", [], True),
    ("인용: 문서 이름 생략", "work", Q0, "결혼 휴가는 6일입니다 [취업규칙 제28조, 제29조].", [], True),
    ("인용: 시행령 생략", "work", Q0, "유급휴일은 개근한 사람에게 줍니다 [근로기준법 제55조, 시행령 제30조].", [], True),
    ("인용 아님: [안내]", "work", Q0, "[안내] 결혼 휴가는 6일입니다 [취업규칙 제28조].", [], True),
    ("인용 아님: [7일] 속 숫자 검사", "work", Q0, "결혼 휴가는 [7일]입니다 [취업규칙 제28조].", [], False),
    ("인용: 받지 않은 조", "work", Q0, "결혼 휴가는 6일입니다 [취업규칙 제99조].", [], False),
    # 결함 9: 버림은 식의 마지막 결과에만
    ("버림을 중간값에 적용 불인정", "welfare", "주 18시간 파트타이머예요",
     "(470,000원 × 18 ÷ 40) + 10,000원 = 221,500원이고 1,000원 미만은 버려 211,000원입니다.", [], False),
]


def check_cases() -> int:
    failed = 0
    for name, draft, cites, expect in CASES:
        r = check_grounding(draft, cites, IDS, EVIDENCE, QUESTION)
        failed += report(name, r, expect)
    for name, cat, question, draft, cites, expect in REAL_CASES:
        chunks = context.assemble([cat])
        r = check_grounding(draft, cites, [c.id for c in chunks], context.render(chunks), question)
        failed += report(name, r, expect)
    return failed


def report(name: str, r: dict, expect: bool) -> int:
    mark = "OK " if r["ok"] == expect else "틀림"
    print(f"[{mark}] {name}: {'통과' if r['ok'] else '실패'} {r['problems']}")
    return int(r["ok"] != expect)


def check_safe_eval() -> int:
    """결함 6: 거듭제곱·긴 식은 계산하지 않고, 오래 걸리지도 않는다."""
    start = time.time()
    results = [agent._eval("2**10", []), agent._eval("9**9**9**9", []),
               agent._eval("+".join(["1"] * 150), []), agent._eval("__import__('os')", [])]
    fast = time.time() - start < 1
    ok = all(r is None for r in results) and fast and agent._eval("(8 * 2) / 4 - -1", [])[0] == 5
    print(f"[{'OK ' if ok else '틀림'}] 안전 계산기: 거듭제곱·긴 식·이름 거부 {results}, {time.time() - start:.3f}초")
    return int(not ok)


def check_verify_node() -> int:
    """결함 8: 버려진 1회차 초안의 uncovered 때문에 붙은 handoff가 2회차 기록에 남지 않는다."""
    chunks = context.assemble(["work"])
    state = {"question": Q0, "evidence_ids": [c.id for c in chunks], "evidence_text": context.render(chunks),
             "lookup_tools": ["lookup_work"], "tools_called": ["lookup_work"], "verify_log": [],
             "draft": "결혼 휴가는 7일입니다 [취업규칙 제28조].", "citations": [], "uncovered": ["경조금"], "attempts": 1}
    state.update(agent.verify(state))
    first = list(state["tools_called"])
    state.update({"draft": "결혼 휴가는 6일입니다 [취업규칙 제28조].", "uncovered": [], "attempts": 2})
    state.update(agent.verify(state))
    second = list(state["tools_called"])
    ok = first == ["lookup_work", "handoff"] and second == ["lookup_work"] and state["verify"]["ok"]
    print(f"[{'OK ' if ok else '틀림'}] verify 도구 기록: 1회차 {first} → 2회차 {second}")
    return int(not ok)


def check_parse_none() -> int:
    """결함 10: 모델이 거절하거나 출력이 잘려 output_parsed가 None이면 분명한 오류를 낸다(가짜 클라이언트, 모델 호출 없음)."""
    resp = SimpleNamespace(output_parsed=None, usage=None, status="incomplete",
                           incomplete_details=SimpleNamespace(reason="max_output_tokens"))
    fake = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kw: resp))
    saved, agent._client = agent._client, fake
    try:
        agent._parse("m", [], agent.Answer, {})
        ok, msg = False, "오류 없음"
    except RuntimeError as e:
        ok, msg = "max_output_tokens" in str(e), str(e)
    finally:
        agent._client = saved
    print(f"[{'OK ' if ok else '틀림'}] 형식 없는 응답: {msg}")
    return int(not ok)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    failed = check_cases() + check_safe_eval() + check_verify_node() + check_parse_none()
    total = len(CASES) + len(REAL_CASES) + 3
    print(f"\n{total - failed}/{total} 맞음")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
