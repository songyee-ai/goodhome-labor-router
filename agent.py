"""라우팅 에이전트 파이프라인 (LangGraph).

classify → gate ─(넘김)→ handoff → END
                └(답변)→ retrieve → answer → verify ─(통과)→ finalize → END
                                        ↑          └(실패, 재시도 1회)┘
                                        └─────────(두 번째도 실패)→ finalize(넘김 문구)

- classify: 질문을 카테고리 하나(+보조 칸)로 고르고 확신도를 낸다. 넘길지는 정하지 않는다.
- gate: 범위 밖이거나 확신도가 기준보다 낮으면 넘긴다. 분류와 넘김 판단을 분리한 노드.
- retrieve: 고른 칸의 조회 도구(lookup_*)만 호출해 그 칸의 근거 조각만 모은다.
- answer: 근거만으로 답하고, 근거에 없는 부분은 uncovered로 돌려준다(부분 넘기기).
- verify: 모델 없이 코드로 인용 id와 숫자의 출처를 검사한다.
"""
import ast
import os
import re
import sys
import time
from functools import lru_cache
from typing import Literal, NamedTuple, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field

import context
import prompts

load_dotenv()

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-5.6-luna")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5.6-terra")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
MAX_ATTEMPTS = 2

CATEGORIES = Literal["work", "pay", "welfare", "contract", "harassment", "out_of_scope"]

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


class Classification(BaseModel):
    category: CATEGORIES
    secondary: Optional[CATEGORIES] = Field(description="보조 칸. 필요 없으면 null")
    confidence: float = Field(description="0~1")
    reason: str


class Answer(BaseModel):
    answer: str
    citations: list[str] = Field(description="답에 실제로 쓴 근거 id")
    uncovered: list[str] = Field(description="질문 중 근거에 없어 답하지 못한 부분. 없으면 빈 목록")


class AnswerStrictHandoff(BaseModel):
    """개선 실험(HANDOFF_STRICT=1): 넘김 판단을 출력 형식의 정의로 좁힌다."""
    answer: str
    citations: list[str] = Field(description="답에 실제로 쓴 근거 id")
    uncovered: list[str] = Field(description=(
        "질문자가 물은 핵심 물음 가운데 근거로 전혀 답할 수 없어 담당 창구로 넘겨야 하는 것만 적는다. "
        "근거로 답은 했지만 개인별 실제 금액, 구체적인 처리 방식 같은 세부가 근거에 없는 정도는 적지 않고, "
        "답변 안에서 '자세한 사항은 담당 창구에 확인'으로 안내한다. 대부분의 질문은 빈 목록이다."))


def _answer_schema() -> type[BaseModel]:
    return AnswerStrictHandoff if os.getenv("HANDOFF_STRICT", "0") == "1" else Answer


class State(TypedDict, total=False):
    question: str
    category: str
    secondary: Optional[str]
    confidence: float
    reason: str
    route: str              # "answer" | "out_of_scope" | "unclear"
    tools_called: list[str]
    lookup_tools: list[str]  # retrieve가 부른 조회 도구. verify가 tools_called를 매번 이것으로 다시 만든다
    evidence_ids: list[str]
    evidence_text: str
    draft: str
    citations: list[str]
    uncovered: list[str]
    attempts: int
    verify: dict            # {"ok": bool, "problems": [...], "checked_numbers": [...]}
    verify_log: list[dict]
    final_answer: str
    usage: dict


def _parse(model: str, messages: list[dict], schema: type[BaseModel], usage: dict):
    resp = client().responses.parse(model=model, input=messages, text_format=schema)
    u = resp.usage
    usage["input_tokens"] = usage.get("input_tokens", 0) + (u.input_tokens if u else 0)
    usage["output_tokens"] = usage.get("output_tokens", 0) + (u.output_tokens if u else 0)
    if resp.output_parsed is None:  # 모델이 거절했거나 출력이 잘려 형식대로 읽을 수 없다
        reason = getattr(getattr(resp, "incomplete_details", None), "reason", None) or getattr(resp, "status", None)
        raise RuntimeError(f"{model} 응답을 {schema.__name__} 형식으로 읽지 못했습니다(거절 또는 출력 잘림: {reason})")
    return resp.output_parsed


# ── 노드 ──────────────────────────────────────────────────────────────

def classify(state: State) -> State:
    usage = dict(state.get("usage", {}))
    c = _parse(ROUTER_MODEL, prompts.classify_messages(state["question"]), Classification, usage)
    secondary = c.secondary if c.secondary not in (None, c.category, "out_of_scope") else None
    return {"category": c.category, "secondary": secondary, "confidence": c.confidence,
            "reason": c.reason, "usage": usage, "tools_called": [], "attempts": 0, "verify_log": []}


def gate(state: State) -> State:
    if state["category"] == "out_of_scope":
        return {"route": "out_of_scope"}
    if state["confidence"] < CONF_THRESHOLD:
        return {"route": "unclear"}
    return {"route": "answer"}


def handoff(state: State) -> State:
    template = prompts.HANDOFF_OUT_OF_SCOPE if state["route"] == "out_of_scope" else prompts.HANDOFF_UNCLEAR
    return {"tools_called": [context.HANDOFF_TOOL], "evidence_ids": [],
            "final_answer": template.format(contact=context.handoff_contact())}


def retrieve(state: State) -> State:
    cats = [state["category"]] + ([state["secondary"]] if state.get("secondary") else [])
    chunks = context.assemble(cats)
    tools = [context.TOOL_OF[c] for c in cats]
    return {"tools_called": tools, "lookup_tools": tools,
            "evidence_ids": [c.id for c in chunks],
            "evidence_text": context.render(chunks)}


def answer(state: State) -> State:
    usage = dict(state.get("usage", {}))
    feedback = None
    if state.get("verify") and not state["verify"]["ok"]:
        feedback = "\n".join(f"- {p}" for p in state["verify"]["problems"])
    msgs = prompts.answer_messages(state["question"], state["evidence_text"],
                                   context.handoff_contact(), feedback)
    a = _parse(ANSWER_MODEL, msgs, _answer_schema(), usage)
    return {"draft": a.answer, "citations": a.citations, "uncovered": a.uncovered,
            "attempts": state.get("attempts", 0) + 1, "usage": usage}


def verify(state: State) -> State:
    result = check_grounding(state["draft"], state["citations"], state["evidence_ids"],
                             state["evidence_text"], state["question"])
    # 초안마다 다시 만든다: 버려진 앞 초안의 uncovered 때문에 붙은 handoff가 남지 않게
    tools = list(state["lookup_tools"]) + ([context.HANDOFF_TOOL] if state.get("uncovered") else [])
    log = state.get("verify_log", []) + [{"attempt": state["attempts"], **result}]
    return {"verify": result, "verify_log": log, "tools_called": tools}


def finalize(state: State) -> State:
    if state["verify"]["ok"]:
        return {"final_answer": state["draft"]}
    tools = list(state["tools_called"])
    if context.HANDOFF_TOOL not in tools:
        tools.append(context.HANDOFF_TOOL)
    return {"tools_called": tools,
            "final_answer": prompts.HANDOFF_UNVERIFIED.format(contact=context.handoff_contact())}


def after_gate(state: State) -> str:
    return "retrieve" if state["route"] == "answer" else "handoff"


def after_verify(state: State) -> str:
    if state["verify"]["ok"] or state["attempts"] >= MAX_ATTEMPTS:
        return "finalize"
    return "answer"


# ── 기계적 검증 ───────────────────────────────────────────────────────
#
# 숫자는 "값 + 단위군"으로 비교한다. 근거의 조·항·호 번호, 원문자, 날짜, 판 표기 같은
# 참조 숫자는 먼저 지워서, 답의 "37일"이 "제37조"의 37로 통과하는 일을 막는다.

CITE = re.compile(r"\[([^\[\]]+)\]")
CITE_ID = re.compile(r"^(.*?)\s*(제\s?\d+\s?조(?:의\s?\d+)?)")
CITE_LABEL = re.compile(r"^(?:근거|출처|참고)\s*[:：]\s*")
REFS = re.compile("|".join([
    r"<근거 id=\"[^\"]*\">", r"</근거>",
    r"\[(?:시행|법률|대통령령|총리령|고용노동부령|고용노동부고시)[^\[\]]*\]",
    r"(?:법률|대통령령|총리령|고용노동부령|고용노동부고시|고시)\s*제\s?[\d\-]+\s?호",
    r"제\s?\d+\s?조(?:의\s?\d+)?",
    r"제\s?\d+(?:-\d+)?\s?(?:항|호|장|절|관|목)",
    r"[①-⑳]",                  # 원문자 ①~⑳
    r"^[ \t]*\d+\.\s",           # 목록 번호 "1. "
    r"https?://\S+",
]), re.M)
DATES = [  # (정규식) 연·월·일 세 칸. 빈 칸은 빈 묶음으로 둔다
    re.compile(r"(?<!\d)(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?"),
    re.compile(r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(?<!\d)(\d{4})\s*년\s*(\d{1,2})\s*월()"),
    re.compile(r"(?<!\d)()(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(?<!\d)((?:19|20)\d\d)\s*년도?()()"),
    re.compile(r"(?<!\d)()(\d{1,2})월(?![급요])()"),    # "N월"만. "월 187,000원"의 '월'은 단위가 아니다
]
RANGE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*([~∼])\s*"
                   r"(?=\d+(?:\.\d+)?\s*(시간|시|근무일|영업일|일|원|%|개월|년|주|회|명|세|분))")
WON_PARTS = re.compile(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만|천)?")
WON_SCALE = {"억": 1e8, "천만": 1e7, "백만": 1e6, "만": 1e4, "천": 1e3}
CLOCK_PM = {"오후", "밤", "저녁"}
TRUNCATION = re.compile(r"버림|버린|버려|절사|미만")
OPERAND_CONSTANTS = {1.0, 12.0, 60.0, 100.0}  # 시간 환산(60), 연·월(12), 비율(100), 그대로(1)
UNIT_CONSTANTS = {("ratio", 100.0), ("minute", 60.0)}  # 식 재료로 쓰는 "× 100%", "÷ 60분"
MAX_EXPR = 200
MAX_VALUE = 1e13
_N = r"(?<![\d.])(\d+(?:\.\d+)?)"


class Qty(NamedTuple):
    group: str          # 단위군: won, minute, ratio, day, month, week, times, people, age, item, clock
    norm: float         # 비교값(원, 분, 퍼센트, 개월 등 단위군 안에서 하나로 맞춘 값)
    math: float         # 식 안에서의 값(쓴 단위 그대로. 금액은 원, %는 /100)
    text: str
    also: tuple = ()    # 근거에서 읽을 때 함께 허용할 비교값("100분의 50" → 150%)


def _won(m):
    v = sum(float(n) * WON_SCALE.get(u, 1) for n, u in WON_PARTS.findall(m.group(0)) if n)
    return "won", v, v


def _clock(m):
    ampm, h, mi, half = m.group(1), int(m.group(2)), m.group(3), m.group(4)
    h += 12 if ampm in CLOCK_PM and h < 12 else 0
    v = h * 60 + (int(mi) if mi else 30 if half else 0)
    return "clock", v, v / 60


def _fraction(m):
    """'100분의 50'(분모가 먼저)을 비율로 읽는다. 100분의 N이면 (100+N)%도 함께 허용한다."""
    b, a = float(m.group(1)), float(m.group(2))
    if not b:
        return None
    also = (100 + a,) if b == 100 else ()
    return "ratio", a / b * 100, a / b, also


def _hours(m):
    h = float(m.group(1)) + (float(m.group(2)) / 60 if m.group(2) else 0)
    return "minute", h * 60, h


def _per(m, group: str, scale: float):
    """'1/3시간'처럼 분수에 단위가 붙은 값."""
    a, b = float(m.group(1)), float(m.group(2))
    return (group, a / b * scale, a / b) if b else None


def _plain(group: str, scale: float = 1):
    return lambda m: (group, float(m.group(1)) * scale, float(m.group(1)))


QUANTITIES = [  # (정규식, 처리) 앞에서부터 차례로 찾고, 찾은 자리는 표시로 바꿔 다음 정규식이 다시 읽지 않게 한다
    (rf"{_N}\s*분의\s*(\d+(?:\.\d+)?)", _fraction),
    (r"(?<![\d.])(?:\d+(?:\.\d+)?\s*(?:억|천만|백만|만|천)\s*)+(?:\d+\s*)?원(?![칙인래])"
     r"|(?<![\d.])\d+(?:\.\d+)?\s*원(?![칙인래])", _won),
    (r"(?<![\d.])()(\d{1,2}):(\d{2})()(?!\d)", _clock),
    (r"(?:(오전|오후|밤|저녁|새벽|낮|아침)\s*)?(?<![\d.])(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분|\s*(반))?(?![간행급점작])",
     _clock),
    (r"(?<![\d.])(\d+)\s*/\s*(\d+)\s*시간", lambda m: _per(m, "minute", 60)),
    (rf"{_N}\s*시간(?:\s*(\d+)\s*분)?", _hours),
    (rf"{_N}\s*분(?![의기])", _plain("minute")),
    (rf"{_N}\s*(?:%|퍼센트)", lambda m: ("ratio", float(m.group(1)), float(m.group(1)) / 100)),
    (rf"{_N}\s*배(?![우정치상송])", _plain("ratio", 100)),
    (rf"{_N}\s*(?:근무일|영업일|일)(?!반)", _plain("day")),
    (rf"{_N}\s*개월", _plain("month")),
    (rf"{_N}\s*년", _plain("month", 12)),
    (rf"{_N}\s*주(?:일|간)?(?![휴년])", _plain("week")),
    (rf"{_N}\s*회(?![사의계식])", _plain("times")),
    (rf"{_N}\s*명(?![절세령단])", _plain("people")),
    (rf"{_N}\s*(?:세|살)(?![대금부])", _plain("age")),
    (rf"{_N}\s*개(?![월인])", _plain("item")),
]
QUANTITIES = [(re.compile(p), h) for p, h in QUANTITIES]
MARK = re.compile("\ue000([\ue001-\ue00a]+)\ue00b")   # 숫자+단위 자리 표시(숫자를 쓰지 않아 다음 정규식에 안 걸린다)
RUN = re.compile(r"[\d\. \t\*/\+\-\(\)=\ue000-\ue00b]+")


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def _mark(i: int) -> str:
    return "\ue000" + "".join(chr(0xe001 + int(d)) for d in str(i)) + "\ue00b"


def _quantities(text: str) -> tuple[str, list[Qty], list[tuple]]:
    """글에서 숫자+단위와 날짜를 뽑는다. 돌려주는 글에서 숫자+단위 자리는 표시로, 날짜 자리는 ';'로 바뀐다.

    조·항·호 번호, 원문자, 판 표기는 먼저 지운다. '09~18시'처럼 범위 앞쪽 숫자에는 뒤의 단위를 붙인다.
    """
    t = REFS.sub(" ", _norm(text).replace("**", ""))
    dates = []
    for d in DATES:
        def keep(m):
            dates.append((tuple(int(x) if x else None for x in m.groups()), m.group(0).strip()))
            return " ; "
        t = d.sub(keep, t)
    t = RANGE.sub(lambda m: m.group(1) + m.group(3) + m.group(2), t)
    tokens: list[Qty] = []
    for pattern, handle in QUANTITIES:
        def put(m):
            r = handle(m)
            if r is None:
                return m.group(0)
            tokens.append(Qty(r[0], r[1], r[2], m.group(0).strip(), r[3] if len(r) > 3 else ()))
            return _mark(len(tokens) - 1)
        t = pattern.sub(put, t)
    return t, tokens, dates


def _numbers_in(masked: str) -> set[float]:
    """단위가 붙지 않은 숫자들(_quantities가 돌려준 글에서)."""
    return {float(x) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", masked)}


@lru_cache(maxsize=64)
def _pool(text: str) -> tuple[dict, frozenset, tuple]:
    """허용 목록: 단위군별 비교값, 식의 재료로 쓸 수 있는 숫자, 날짜."""
    masked, tokens, dates = _quantities(text)
    units: dict[str, set] = {}
    bare = _numbers_in(masked)
    for q in tokens:
        vals = (q.norm,) + q.also
        units.setdefault(q.group, set()).update(vals)
        bare |= {q.math, q.norm}
        if q.group == "ratio":
            bare |= {v / 100 for v in vals}   # "100분의 60" → × 0.6, × 1.6
    return units, frozenset(bare), tuple(d for d, _ in dates)


def _math_view(masked: str) -> str:
    """계산식을 읽기 쉽게 바꾼다: ×·÷·x는 연산자로, 연산자와 숫자 사이 낱말은 지우고, 문장 끝·줄바꿈에서 식을 자른다."""
    t = masked.replace("×", "*").replace("✕", "*").replace("÷", "/").replace("−", "-")
    t = re.sub(r"(?<=[\d\)\ue00b])\s*[xX]\s*(?=[\d\(\ue000])", "*", t)
    t = re.sub(r"(?<=[\(\*/\+\-=])\s*(?:[가-힣]+\s*){1,2}(?=[\d\(\ue000])", " ", t)   # "× 주 3일" → "× 3일"
    t = re.sub(r"(?<=[\d\)\ue00b])\s*(?:[가-힣]+\s*){1,2}(?=[\*/\+\-=])", " ", t)      # "18시간분 ÷" → "18시간 ÷"
    t = re.sub(r"\.(?!\d)|\n", " ; ", t)
    return re.sub(r"(?<![\d.])0+(?=\d)", "", t)


_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def _eval(expr: str, tokens: list[Qty]) -> Optional[tuple[float, list]]:
    """사칙연산·괄호·부호만 계산한다. 값과 재료(단위 없는 숫자 또는 Qty) 목록을 돌려준다."""
    expr = MARK.sub(lambda m: "q" + "".join(str(ord(c) - 0xe001) for c in m.group(1)), expr).strip()
    if not expr or len(expr) > MAX_EXPR:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    leaves = []

    def ev(node) -> float:
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            v = _OPS[type(node.op)](ev(node.left), ev(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            v = float(node.value)
            leaves.append(v)
        elif isinstance(node, ast.Name) and node.id[1:].isdigit() and int(node.id[1:]) < len(tokens):
            q = tokens[int(node.id[1:])]
            v = q.math
            leaves.append(q)
        else:
            raise ValueError(expr)
        if abs(v) > MAX_VALUE:
            raise ValueError(expr)
        return v

    try:
        return ev(tree.body), leaves
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _paren_groups(expr: str) -> list[str]:
    """짝이 맞는 괄호 안의 식을 안쪽 괄호부터 바깥 괄호까지 모두 꺼낸다."""
    groups, stack = [], []
    for i, ch in enumerate(expr):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            groups.append(expr[start + 1:i])
    return groups


def _balanced(expr: str) -> str:
    """짝이 없는 괄호를 지운다. '(2일 + 8일 = 10일)'을 등호로 자르면 양쪽에 괄호가 하나씩 남는다."""
    keep, stack = list(expr), []
    for i, ch in enumerate(expr):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                keep[i] = " "
    for i in stack:
        keep[i] = " "
    return "".join(keep)


def _computed_numbers(masked: str, tokens: list[Qty], is_known) -> tuple[set[float], set[Qty]]:
    """답변 안의 계산식에서 직접 계산한 값들.

    '(8 × 20,000원 × 50%) + (2 × 20,000원 × 100%) = 80,000원 + 40,000원 = 120,000원'처럼
    괄호와 이어진 등호를 처리한다. '주 2일 × 하루 7시간이면 14시간'처럼 등호 없이 쓴 식도 계산한다.
    식은 앞에서부터 차례로 읽고, 식의 재료가 모두 근거·질문의 숫자이거나 앞선 식의 결과일 때만
    그 값을 인정한다(is_known으로 판단). '2일 + 8일 = 10일'처럼 등식이 맞고, 모르는 재료가 하나뿐이며,
    결과 쪽이 근거·질문의 숫자이면 그 모르는 재료도 계산값으로 인정한다(결과에서 거꾸로 푼 식).
    값을 버림한 경우는 같은 문장 안, 식 바로 뒤에 '버림·미만·절사'라는 말이 있을 때 그 식의 마지막 결과에만 인정한다.
    계산값과 함께, 인정된 식에 재료로 쓰인 숫자+단위(예: '× 100%')도 돌려준다.
    """
    t = _math_view(masked)
    values: set[float] = set()
    used: set[Qty] = set()
    for m in RUN.finditer(t):
        span = m.group(0)
        if not re.search(r"[\*/\+\-]", span.strip().lstrip("-+")):
            continue
        last, sides = None, []
        for side in span.split("="):
            side = _balanced(side)
            whole = _eval(side, tokens)
            if whole is not None:
                sides.append(whole)
            for g in _paren_groups(side) + [side]:
                if not re.search(r"[\*/\+\-]", g.strip().lstrip("-+")):
                    continue
                r = _eval(g, tokens)
                if r is None or not all(is_known(x, values) for x in r[1]):
                    continue
                values.add(r[0])
                used |= {x for x in r[1] if isinstance(x, Qty)}
                last = r[0]
        unknown = [x for _, leaves in sides for x in leaves if not is_known(x, values)]
        if (len(sides) >= 2 and len(unknown) == 1 and all(abs(v - sides[0][0]) < 0.005 for v, _ in sides)
                and any(len(leaves) == 1 and is_known(leaves[0], values) for _, leaves in sides)):
            x = unknown[0]
            values.add(x.math if isinstance(x, Qty) else x)
            used |= {x} if isinstance(x, Qty) else set()
        if last is not None and TRUNCATION.search(t[m.end():].split(";")[0][:40]):
            values |= {float(int(last // 10**k) * 10**k) for k in range(1, 4)}
    return values, used


def _grounded(n: float, pool) -> bool:
    return any(abs(n - p) < 0.005 for p in pool)


def _date_grounded(d: tuple, pool) -> bool:
    """답의 날짜에 적힌 칸(연·월·일)이 모두 근거·질문의 어느 날짜와 같으면 인정한다."""
    return any(all(a is None or a == b for a, b in zip(d, p)) for p in pool)


def _cited(inner: str, ids: set[str]) -> tuple[list[str], list[str]]:
    """대괄호 안을 (인용한 근거 id, 인용이 아닌 조각)으로 나눈다.

    '[취업규칙 제28조 제1항]'은 조까지 줄여 읽고, '[취업규칙 제28조, 제29조]'처럼 문서 이름이 빠진
    조각에는 앞 조각의 문서 이름을 붙인다. '2026년 적용 최저임금'처럼 조문이 아닌 근거 id도 인용으로 본다.
    """
    cited, rest, doc = [], [], ""
    for part in re.split(r"[,;·/]|\s및\s", inner):
        p = CITE_LABEL.sub("", part.strip().strip("「」『』\"'"))
        p = p.replace("「", "").replace("」", "")
        if not p:
            continue
        if p in ids:
            cited.append(p)
            m = CITE_ID.match(p)
            doc = m.group(1).strip() if m and m.group(1).strip() else doc
            continue
        m = CITE_ID.match(p)
        if not m:
            rest.append(part)
            continue
        head, art = m.group(1).strip(), re.sub(r"\s", "", m.group(2))
        tries = [f"{head} {art}"] if head else []
        if doc:
            tries += [f"{doc} {head} {art}", f"{doc} {art}"] if head else [f"{doc} {art}"]
        if not head and not doc:
            tries += sorted(i for i in ids if i.endswith(" " + art))
        cid = next((c for c in tries if c in ids), tries[0] if tries else art)
        cited.append(cid)
        mm = CITE_ID.match(cid)
        doc = mm.group(1).strip() if mm and mm.group(1).strip() else doc
    return cited, rest


def check_grounding(draft: str, citations: list[str], evidence_ids: list[str],
                    evidence_text: str, question: str) -> dict:
    problems = []
    ids = set(evidence_ids)

    cited = []
    for c in citations:
        got, rest = _cited(c, ids)
        cited += got or [c.strip()]

    def strip_cite(m):
        got, rest = _cited(m.group(1), ids)
        cited.extend(got)
        return (" " + ", ".join(rest) + " ") if got else m.group(0)  # 인용이 아닌 대괄호는 숫자 검사에 남긴다

    body = CITE.sub(strip_cite, draft)
    bad_ids = sorted({c for c in cited if c not in ids})
    if bad_ids:
        problems.append(f"받은 근거에 없는 id를 인용함: {', '.join(bad_ids)}")

    ev_units, ev_bare, ev_dates = _pool(evidence_text + "\n" + context.handoff_contact())
    q_units, q_bare, q_dates = _pool(question)

    clocks = ev_units.get("clock", set()) | q_units.get("clock", set())
    spans = ev_units.get("minute", set()) | q_units.get("minute", set()) | {0.0}

    def in_units(q: Qty) -> bool:
        if q.group == "clock":
            # 시각은 12시간제로 쓰는 일이 많아 12시간 차이를 같게 보고, '10시 출근 + 9시간 = 19시'처럼
            # 근거·질문의 시각에 근거·질문의 시간 하나를 더하거나 뺀 시각까지 인정한다
            return any(round(q.norm - c - s * k) % 720 == 0 for c in clocks for s in spans for k in (1, -1))
        return _grounded(q.norm, ev_units.get(q.group, ())) or _grounded(q.norm, q_units.get(q.group, ()))

    def is_known(x, results) -> bool:
        """식의 재료: 근거·질문의 숫자(단위가 붙었으면 같은 단위군), 앞선 식의 결과, 계산용 상수."""
        if isinstance(x, Qty):
            return (in_units(x) or _grounded(x.math, results | q_bare)
                    or (x.group, x.norm) in UNIT_CONSTANTS)
        return _grounded(x, ev_bare | q_bare | OPERAND_CONSTANTS | results)

    masked, tokens, dates = _quantities(body)
    computed, operands = _computed_numbers(masked, tokens, is_known)
    checked, ungrounded = [], []
    for q in tokens:
        checked.append(q.text)
        if not (q in operands or in_units(q) or _grounded(q.math, computed | q_bare)):
            ungrounded.append(q.text)
    for d, text in dates:
        checked.append(text)
        if not _date_grounded(d, ev_dates + q_dates):
            ungrounded.append(text)
    if ungrounded:
        problems.append("근거·질문에 없고 계산식으로도 나오지 않은 숫자: " + ", ".join(dict.fromkeys(ungrounded))
                        + " — 계산해서 얻은 숫자면 '10시간 − 8시간 = 2시간'처럼 식을 함께 쓰고, 근거에 없는 숫자면 지우세요")
    return {"ok": not problems, "problems": problems, "checked_numbers": checked}


# ── 그래프 ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("gate", gate)
    g.add_node("handoff", handoff)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("verify", verify)
    g.add_node("finalize", finalize)
    g.add_edge(START, "classify")
    g.add_edge("classify", "gate")
    g.add_conditional_edges("gate", after_gate, {"retrieve": "retrieve", "handoff": "handoff"})
    g.add_edge("handoff", END)
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_verify, {"answer": "answer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = build_graph()


def run(question: str) -> State:
    start = time.time()
    state = GRAPH.invoke({"question": question})
    state["seconds"] = round(time.time() - start, 2)
    return state


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    q = " ".join(sys.argv[1:]) or "결혼하면 휴가 며칠이에요?"
    s = run(q)
    print("질문:", q)
    print(f"분류: {s['category']} / 보조 {s.get('secondary')} / 확신도 {s['confidence']:.2f} → {s['route']}")
    print("이유:", s["reason"])
    print("호출한 도구:", s["tools_called"])
    if s.get("verify_log"):
        for v in s["verify_log"]:
            print(f"검증 {v['attempt']}회차: {'통과' if v['ok'] else '실패'} {v['problems']}")
    print("인용:", s.get("citations"))
    print("근거에 없음:", s.get("uncovered"))
    print("\n" + s["final_answer"])
    print(f"\n({s['seconds']}초, 토큰 {s.get('usage')})")
