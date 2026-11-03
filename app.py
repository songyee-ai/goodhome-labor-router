"""굿홈앤리빙 노동 문의 안내 데모 화면 (streamlit).

실행: .venv\\Scripts\\python.exe -m streamlit run app.py

질문 하나를 agent.run()으로 처리하고, 답변과 함께
분류·넘김 판단·호출한 도구, 근거로 쓴 조각, 코드 검증 결과를 보여 준다.
새 질문 예시는 data/examples.json에서 가져오며, 저장된 평가 기록은 별도 재생 화면으로 보여 준다.
"""
import json
import os
import re
from pathlib import Path

import streamlit as st

os.environ.setdefault("STRICT_SECONDARY", "1")

import agent
import context

ROOT = Path(__file__).resolve().parent
EXAMPLE_IDS = ["E01", "E03", "E04", "E07", "E08", "E10"]  # 칸마다 하나씩 + 범위 밖 + 두 칸 복합

st.set_page_config(page_title="굿홈앤리빙 노동 문의 안내", layout="wide")


# ── 준비 ──────────────────────────────────────────────────────────────

@st.cache_data
def load_examples() -> list[dict]:
    items = json.loads((ROOT / "data" / "examples.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in items}
    picked = [by_id[i] for i in EXAMPLE_IDS if i in by_id]
    return picked or items[:6]


def category_name(key) -> str:
    if not key:
        return "없음"
    info = context.category_map().get(key, {})
    return f"{info.get('name', key)} ({key})"


def category_guide() -> list[tuple[str, str, str]]:
    """prompts.CATEGORY_GUIDE의 '- key (이름): 설명' 줄을 (key, 이름, 설명)으로 나눈다."""
    import prompts
    rows = []
    for line in prompts.CATEGORY_GUIDE.splitlines():
        m = re.match(r"-\s*(\w+)\s*\(([^)]*)\):\s*(.*)", line)
        if m:
            rows.append(m.groups())
    return rows


def md_safe(text: str) -> str:
    """'2041~2043'의 물결표나 금액의 $가 취소선·수식으로 그려지지 않게 막는다."""
    return (text or "").replace("~", r"\~").replace("$", r"\$")


def md_lines(text: str) -> str:
    """조문 원문의 줄바꿈을 그대로 보이게 한다(표 줄은 그대로 둔다)."""
    lines = md_safe(text).split("\n")
    return "\n".join(l if l.lstrip().startswith("|") or not l.strip() else l + "  " for l in lines)


def hide_secrets(msg: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_\-\*\.]+", "sk-…(가림)", msg)


def has_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


# ── 결과 읽기 ─────────────────────────────────────────────────────────

def outcome(s: dict) -> tuple[str, str]:
    """(종류, 안내 문구). 종류: answer | partial | handoff_scope | handoff_unclear | handoff_unverified"""
    route = s.get("route")
    if route == "out_of_scope":
        return "handoff_scope", "사내 규정·안내 범위의 법령에 근거가 없는 질문이라 담당 창구로 넘겼습니다."
    if route == "unclear":
        return "handoff_unclear", "무엇을 묻는지 분류 확신도가 낮아 질문을 되묻고 담당 창구를 안내했습니다."
    log = s.get("verify_log") or []
    if log and not log[-1].get("ok"):
        return "handoff_unverified", "답변이 코드 검증을 통과하지 못해, 답변 대신 담당 창구로 넘겼습니다."
    if s.get("uncovered"):
        return "partial", "질문 중 근거에 없는 부분은 지어내지 않고 담당 창구로 넘겼습니다."
    return "answer", ""


def evidence_chunks(s: dict) -> dict:
    """이 질문에서 조회한 근거 조각 id → Chunk."""
    if s.get("route") != "answer" or not s.get("category"):
        return {}
    cats = [c for c in [s.get("category"), s.get("secondary")] if c and c in context.TOOL_OF]
    return {c.id: c for c in context.assemble(cats)}


def cited_ids(s: dict, chunks: dict) -> list[str]:
    """모델이 낸 citations + 답변 본문의 [근거 id] 표시(받은 근거에 있는 것만)."""
    ids = list(dict.fromkeys(s.get("citations") or []))
    for inner in re.findall(r"\[([^\[\]]+)\]", s.get("final_answer") or ""):
        for part in re.split(r"[,;·/]|\s및\s", inner):
            p = part.strip()
            if p in chunks and p not in ids:
                ids.append(p)
    return ids


def node_path(s: dict) -> str:
    if s.get("route") != "answer":
        return "분류(classify) → 넘김 판단(gate) → 넘김(handoff)"
    steps = ["분류(classify)", "넘김 판단(gate)", "근거 조회(retrieve)"]
    for i, _ in enumerate(s.get("verify_log") or [None]):
        steps += [f"답변 작성(answer){'' if i == 0 else ' 다시'}", "검증(verify)"]
    steps.append("마무리(finalize)")
    return " → ".join(steps)


# ── 그리기 ────────────────────────────────────────────────────────────

def tool_badges(tools: list[str]) -> str:
    if not tools:
        return "없음"
    out = []
    for t in tools:
        color = "orange" if t == context.HANDOFF_TOOL else "blue"
        out.append(f":{color}-badge[{t}]")
    return " ".join(out)


def show_route(s: dict):
    st.markdown("**1. 분류**")
    conf = s.get("confidence")
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "알 수 없음"
    st.markdown(
        f"- 주 칸: {category_name(s.get('category'))}\n"
        f"- 보조 칸: {category_name(s.get('secondary'))}\n"
        f"- 확신도: {conf_txt}"
    )
    if s.get("reason"):
        st.caption(f"분류 이유: {md_safe(s['reason'])}")

    st.markdown("**2. 넘김 판단(gate)**")
    route = s.get("route")
    th = agent.CONF_THRESHOLD
    if route == "out_of_scope":
        st.markdown("주 칸이 범위 밖(out_of_scope) → **넘김**")
    elif route == "unclear":
        st.markdown(f"확신도 {conf_txt} < 기준값 {th} → **넘김**(되묻기)")
    else:
        st.markdown(f"범위 안이고 확신도 {conf_txt} ≥ 기준값 {th} → **답변 진행**")

    st.markdown("**3. 호출한 도구**")
    st.markdown(tool_badges(s.get("tools_called") or []))
    st.caption("처리 순서: " + node_path(s))


def show_evidence(s: dict):
    chunks = evidence_chunks(s)
    if not chunks:
        st.info("넘김 경로라 근거 문서를 조회하지 않았습니다.")
        return
    cited = cited_ids(s, chunks)
    used = [c for c in cited if c in chunks]
    missing = [c for c in cited if c not in chunks]

    st.markdown(f"**답변이 인용한 근거 {len(used)}개** (펼치면 원문)")
    if not used:
        st.caption("인용한 근거가 없습니다.")
    for cid in used:
        ch = chunks[cid]
        name = re.search(r"\(([^)]*)\)", ch.title)
        label = cid + (f" · {name.group(1)}" if name else "")
        with st.expander(md_safe(label)):
            st.caption(f"출처: `{ch.source}`")
            st.markdown(md_lines(ch.text))
    if missing:
        st.warning("받은 근거에 없는 id를 인용함: " + ", ".join(missing))

    rest = [cid for cid in chunks if cid not in used]
    st.markdown(f"**받았지만 인용하지 않은 근거 {len(rest)}개**")
    by_source: dict[str, list[str]] = {}
    for cid in rest:
        by_source.setdefault(chunks[cid].source, []).append(cid)
    for src, ids in by_source.items():
        st.caption(f"`{src}` — " + md_safe(", ".join(ids)))


def show_verify(s: dict):
    log = s.get("verify_log") or []
    if not log:
        st.info("넘김 경로라 답변 검증을 하지 않았습니다.")
        return
    if len(log) > 1:
        st.warning(f"1회차 검증에 걸려 문제 목록을 돌려주고 답변을 다시 썼습니다(총 {len(log)}회 작성).")
    for v in log:
        ok = v.get("ok")
        mark = ":green-badge[통과]" if ok else ":red-badge[실패]"
        st.markdown(f"**{v.get('attempt', '?')}회차** {mark}")
        for p in v.get("problems") or []:
            st.markdown(f"- {md_safe(p)}")
        nums = v.get("checked_numbers") or []
        if nums:
            st.caption("검사한 숫자: " + md_safe(", ".join(dict.fromkeys(nums))))
        else:
            st.caption("검사한 숫자: 없음")
    st.caption("검증은 모델 없이 코드로 합니다: 인용한 id가 받은 근거에 있는지, "
               "답변의 숫자가 근거·질문에 있거나 답변 속 계산식으로 나오는지.")

    unc = s.get("uncovered") or []
    st.markdown(f"**근거에 없어 넘긴 부분 {len(unc)}개**")
    for u in unc:
        st.markdown(f"- {md_safe(u)}")


def show_record(rec: dict):
    with st.container(border=True):
        st.markdown(f"**질문** · {md_safe(rec['question'])}")
        if rec.get("error"):
            st.error(f"처리 중 오류가 났습니다: {rec['error']}")
            return
        s = rec["state"]
        kind, note = outcome(s)
        if kind.startswith("handoff"):
            st.warning("**담당 창구로 넘김** — " + note)
        elif kind == "partial":
            st.info("**일부 넘김** — " + note)

        st.markdown(md_safe(s.get("final_answer") or "(답변 없음)"))

        usage = s.get("usage") or {}
        log = s.get("verify_log") or []
        vtxt = "검증 없음(넘김)" if not log else ("검증 통과" if log[-1].get("ok") else "검증 실패")
        st.markdown(
            tool_badges(s.get("tools_called") or []) + "  "
            + f" · {vtxt}"
            + f" · {s.get('seconds', '?')}초"
            + f" · 입력 토큰 {usage.get('input_tokens', 0):,} / 출력 토큰 {usage.get('output_tokens', 0):,}"
        )

        t1, t2, t3 = st.tabs(["처리 경로", "근거", "검증"])
        with t1:
            show_route(s)
        with t2:
            show_evidence(s)
        with t3:
            show_verify(s)


def ask(question: str):
    try:
        with st.spinner("분류하고 근거를 찾아 답하는 중입니다…"):
            state = agent.run(question)
        st.session_state.history.insert(0, {"question": question, "state": dict(state)})
    except Exception as e:  # 모델 호출 실패 등
        msg = hide_secrets(f"{type(e).__name__}: {e}")
        st.session_state.history.insert(0, {"question": question, "error": msg})


# ── 화면 ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def pick_example(q: str):
    st.session_state.pending = q


with st.sidebar:
    mode = st.radio("이용 방식", ["저장된 실행 기록 보기", "새 질문 보내기"])
    st.subheader("개선 스위치")
    for switch in ("STRICT_SECONDARY", "HANDOFF_STRICT", "ANSWER_RULES_V2", "BOUNDARY_EXAMPLES"):
        st.caption(f"{switch}: {'켜짐' if os.getenv(switch, '0') == '1' else '꺼짐'}")
    st.header("이 도구는")
    st.write("가상 회사 **굿홈앤리빙**(홈패브릭·생활용품) 직원의 노동 문의를 담당 칸으로 나누고, "
             "그 칸의 사규·법령 근거만으로 답한 뒤 답변을 코드로 검증합니다.")
    st.caption("회사와 사규는 실존하지 않는 가상의 것입니다.")

    st.subheader("칸 목록")
    for key, name, desc in category_guide():
        tool = context.TOOL_OF.get(key, context.HANDOFF_TOOL)
        with st.expander(f"{name} · `{key}`"):
            st.caption(f"도구: `{tool}`")
            st.write(desc)

    st.subheader("사용 모델")
    st.markdown(f"- 분류: `{agent.ROUTER_MODEL}`\n- 답변: `{agent.ANSWER_MODEL}`\n"
                f"- 넘김 기준 확신도: {agent.CONF_THRESHOLD}")

    st.info("답변은 회사 사규와 일부 노동 법령 근거 범위 안에서만 하며, 법률 자문이 아닙니다. "
            "정확한 판단은 경영지원팀이나 고용노동부 고객상담센터(1350)에 확인하세요.")
    if st.button("대화 지우기", key="clear"):
        st.session_state.history = []

st.title("굿홈앤리빙 노동 문의 안내")
st.caption("질문하면 칸 분류 → 넘김 판단 → 그 칸의 근거 조회 → 답변 → 코드 검증 순서로 처리합니다.")

if mode == "저장된 실행 기록 보기":
    st.info("평가 기록 재생 · 저장된 답변과 당시 검증 결과를 표시합니다. 모델을 호출하지 않습니다.")
    st.caption("기록 설정: STRICT_SECONDARY=1. 사이드바 스위치는 새 질문에 적용할 현재 설정입니다. 당시 검증 결과는 현재 검증기로 다시 판정한 결과가 아닙니다.")
    path = ROOT / "runs" / "05_strict_secondary_r1" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = st.selectbox("문항 선택", range(len(rows)), format_func=lambda i: f"{rows[i]['id']} · {rows[i].get('question', '오류 기록')}")
    row = rows[index]
    state = dict(row)
    state.update(category=row.get("pred_category"), secondary=row.get("pred_secondary"), final_answer=row.get("answer", ""))
    show_record({"question": row.get("question", row["id"]), "state": state, "error": row.get("error")})
else:
    key_ok = has_key()
    if not key_ok:
        st.info("모델을 부를 API 키가 없습니다. 환경변수 `OPENAI_API_KEY`를 설정한 뒤 다시 실행해 주세요.")

    st.markdown("**예시 질문**")
    examples = load_examples()
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        cols[i % 3].button(ex["question"], key=f"ex_{ex['id']}", width="stretch",
                           on_click=pick_example, args=(ex["question"],), disabled=not key_ok)

    with st.form("ask_form", clear_on_submit=True):
        text = st.text_area("질문", placeholder="예: 연차는 언제부터 생겨요?", height=90)
        sent = st.form_submit_button("보내기", type="primary", disabled=not key_ok)
    if sent and text.strip():
        st.session_state.pending = text.strip()

    if st.session_state.pending and key_ok:
        q, st.session_state.pending = st.session_state.pending, None
        ask(q)

    if st.session_state.history:
        st.subheader("대화 기록 (최근 것이 위)")
        for rec in st.session_state.history:
            show_record(rec)
