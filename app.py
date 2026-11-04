"""굿홈앤리빙 노동 문의 안내 데모 화면 (streamlit).

실행: .venv\\Scripts\\python.exe -m streamlit run app.py

질문 하나를 agent.run()으로 처리하고, 답변과 함께
분류·넘김 판단·호출한 도구, 근거로 쓴 조각, 코드 검증 결과를 보여 준다.
새 질문 예시는 data/examples.json에서 가져오며, 저장된 평가 기록은 별도 재생 화면으로 보여 준다.
"""
import json
from html import escape
import os
import re
from pathlib import Path

import streamlit as st

os.environ.setdefault("STRICT_SECONDARY", "1")

import agent
import context

ROOT = Path(__file__).resolve().parent
EXAMPLE_IDS = ["E01", "E03", "E04", "E07", "E08", "E10"]  # 칸마다 하나씩 + 범위 밖 + 두 칸 복합

BRAND_MARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 54 54" role="img" aria-label="굿홈앤리빙 심벌"><defs><mask id="weave"><path fill="white" d="M4.32 21.6 27 3.78 49.68 21.6v25.92L42.12 51.84H4.32Z"/><g fill="black"><rect x="15.12" y="24.84" width="6.48" height="17.28" rx=".54"/><rect x="29.16" y="18.9" width="6.48" height="23.22" rx=".54"/><rect x="15.12" y="31.32" width="26.46" height="5.4" rx=".54"/></g></mask></defs><rect width="54" height="54" fill="#656A83" mask="url(#weave)"/></svg>"""

st.set_page_config(page_title="굿홈앤리빙 · 구성원 안내", page_icon=BRAND_MARK_SVG, layout="wide")

st.markdown("""
<style>
:root { --gh-primary: #656A83; --gh-ink: #45475B; --gh-paper: #FFFCF5; --gh-butter: #FFF0A6; --gh-lilac: #EFEDF6; --gh-mist: #DCE9F0; --gh-muted: #626575; --gh-line: #E3E1E4; }
.stApp { background: var(--gh-paper); color: var(--gh-ink); color-scheme: light; font-family: "Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1160px; padding-top: 2.3rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] { background: var(--gh-lilac); color: var(--gh-ink); border-right: 1px solid var(--gh-line); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.6rem; }
[data-testid="stSidebar"] h3 { font-size: 1rem; color: var(--gh-ink); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { line-height: 1.85; }
h1, h2, h3 { color: var(--gh-ink); letter-spacing: -.045em; }
[data-testid="stCaptionContainer"] { color: var(--gh-muted); }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { background-color: var(--gh-primary); border-color: var(--gh-primary); }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 18px; border-color: var(--gh-line); background: #fff; }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--gh-line); border-radius: 14px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: var(--gh-muted); font-size: .8rem; }
[data-testid="stMetricValue"] { color: var(--gh-ink); font-size: 1.25rem; font-weight: 650; letter-spacing: -.03em; }
[data-baseweb="select"] > div { border-radius: 12px; border-color: var(--gh-line); background: #fff; color: var(--gh-ink); min-height: 48px; }
[data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--gh-line); }
button[data-baseweb="tab"] { padding: 12px 2px; font-weight: 600; color: var(--gh-muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--gh-ink); }
[data-baseweb="tab-highlight"] { background-color: var(--gh-primary); height: 3px; }
[data-testid="stExpander"] { border-color: var(--gh-line); border-radius: 12px; background: #F1F6F8; }
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button { border-radius: 10px; }
[data-testid="stFormSubmitButton"] button[kind^="primary"] { background: var(--gh-primary); border-color: var(--gh-primary); color: #fff; }
.gh-brand { display: flex; align-items: center; gap: 11px; margin-bottom: 28px; color: var(--gh-ink); font-weight: 750; font-size: 1.1rem; letter-spacing: -.04em; }
.gh-logo { display: block; flex: 0 0 40px; width: 40px; height: 40px; }
.gh-logo svg { display: block; width: 40px; height: 40px; }
.gh-brand small { display: block; font-size: .7rem; letter-spacing: .1em; color: var(--gh-muted); font-weight: 500; margin-top: 3px; }
.gh-hero { border: 1px solid #E9DDAB; border-radius: 22px; padding: 32px 36px; background: var(--gh-butter); margin-bottom: 24px; }
.gh-eyebrow { font-size: .75rem; color: var(--gh-ink); letter-spacing: .13em; font-weight: 700; margin-bottom: 10px; }
.gh-hero h1 { font-size: clamp(1.8rem, 3.4vw, 2.65rem); line-height: 1.28; margin: 0 0 13px; padding: 0; color: var(--gh-ink); font-weight: 750; }
.gh-hero p { margin: 0; color: var(--gh-ink); font-size: .96rem; }
.gh-pill { display: inline-block; border-radius: 999px; padding: 5px 11px; font-size: .73rem; font-weight: 650; color: var(--gh-ink); border: 1px solid #DAD9DD; background: var(--gh-paper); margin-top: 18px; }
.gh-section { font-size: 1.05rem; font-weight: 700; color: var(--gh-ink); letter-spacing: -.035em; margin: 10px 0 5px; }
.gh-question { padding: 20px 24px; border-radius: 12px; background: var(--gh-lilac); border-left: 3px solid var(--gh-primary); margin-bottom: 20px; color: var(--gh-ink); font-size: 1.04rem; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }
.gh-label { font-size: .72rem; font-weight: 700; letter-spacing: .1em; color: var(--gh-muted); margin: 3px 0 10px; }
.gh-footer { margin-top: 24px; padding-top: 15px; border-top: 1px solid var(--gh-line); color: var(--gh-muted); font-size: .76rem; line-height: 1.7; }
@media (max-width: 640px) { .gh-hero { padding: 24px 20px; } [data-testid="stMainBlockContainer"] { padding-top: 1.3rem; } .gh-question { padding: 16px; } }

/* 시스템 테마와 관계없이 확정한 밝은 브랜드 화면의 글자·입력·메뉴를 함께 지정한다. */
.stApp, [data-baseweb="popover"], [data-baseweb="modal"], [data-testid="stDialog"] {
  --gh-primary: #656A83; --gh-ink: #45475B; --gh-paper: #FFFCF5;
  --gh-muted: #626575; --gh-lilac: #EFEDF6; --gh-mist: #DCE9F0; --gh-line: #E3E1E4;
  color-scheme: light;
}
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stMarkdownContainer"] :is(p, li, strong, em),
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stRadioOption"],
.stApp [data-testid="stRadioOption"] p,
.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary p,
.stApp [data-testid="stMetricLabel"] p {
  color: var(--gh-ink) !important;
}
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p { color: var(--gh-muted) !important; }
.stApp [data-testid="stRadioOption"] > div > div:first-child {
  background-color: #fff; border: 1px solid var(--gh-primary);
}
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child {
  background-color: var(--gh-primary);
}
.stApp [data-testid="stRadioOption"] > div > div:first-child > div { background-color: #fff; }
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] div {
  color: var(--gh-ink) !important;
}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
.stApp [data-testid="stTextArea"] [data-baseweb="textarea"],
.stApp [data-testid="stTextArea"] textarea,
.stApp [data-testid="stTextInput"] [data-baseweb="input"],
.stApp [data-testid="stTextInput"] input {
  background: #fff !important; color: var(--gh-ink) !important;
  -webkit-text-fill-color: var(--gh-ink);
  border-color: #969AAD !important;
}
.stApp :is(textarea, input)::placeholder {
  color: var(--gh-muted) !important; -webkit-text-fill-color: var(--gh-muted); opacity: 1;
}
.stApp [data-testid="stSelectbox"] svg { color: var(--gh-ink); fill: currentColor; }
.stApp [data-testid="stForm"] { border-color: var(--gh-line); background: #fff; }
.stApp [data-testid="stButton"] button,
.stApp [data-testid="stFormSubmitButton"] button {
  color: var(--gh-ink) !important; background: #fff !important; border-color: #969AAD !important;
}
.stApp [data-testid="stButton"] button p,
.stApp [data-testid="stFormSubmitButton"] button p { color: inherit !important; }
.stApp [data-testid="stFormSubmitButton"] button[kind^="primary"] {
  background: var(--gh-primary) !important; color: #fff !important;
}
.stApp :is([data-testid="stButton"], [data-testid="stFormSubmitButton"]) button:disabled {
  background: #EFEDF6 !important; color: #626575 !important; opacity: 1;
}
.stApp button:focus-visible, .stApp input:focus-visible,
.stApp textarea:focus-visible, .stApp summary:focus-visible {
  outline: 2px solid var(--gh-primary); outline-offset: 3px;
}
.stApp [data-testid="stMarkdownContainer"] code {
  color: var(--gh-ink) !important; background: #EFEDF6 !important;
}
.stApp [data-testid="stMarkdownContainer"] a { color: #4F5576; text-decoration: underline; }
.stApp [data-testid="stAlertContainer"] {
  background: #EAF0F5 !important; color: var(--gh-ink) !important; border: 1px solid #CAD6E0;
}
.stApp [data-testid="stAlertContainer"] svg { color: #4F5576; }
.stApp [data-testid="stMarkdownContainer"] :is(.st-badge, .stMarkdownBadge) {
  background: #EFEDF6 !important; color: var(--gh-ink) !important;
}
.stApp hr { border-color: var(--gh-line); }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="menu"],
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="modal"] [role="dialog"] { background: #fff !important; color: #45475B !important; }
[data-baseweb="popover"] :is(li, [role="option"], [role="menuitem"], button),
[data-baseweb="popover"] :is(li, [role="option"], [role="menuitem"]) :is(div, span, p) {
  color: #45475B !important; background-color: #fff !important;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] :is(div, span) {
  background-color: #EFEDF6 !important; color: #45475B !important;
}


/* 현재 Streamlit의 React Aria 선택창·탭·포털 메뉴도 같은 팔레트로 처리한다. */
.stApp [data-testid="stSelectbox"] [role="group"],
.stApp [data-testid="stSelectbox"] input,
.stApp [data-testid="stSelectbox"] button {
  background: #fff !important; color: #45475B !important;
  -webkit-text-fill-color: #45475B;
}
.stApp [data-testid="stSelectbox"] [role="group"] { border: 1px solid #969AAD; border-radius: 12px; }
.stApp [data-testid="stTabs"] [role="tablist"] { gap: 24px; border-bottom: 1px solid #E3E1E4; }
.stApp [data-testid="stTabs"] [role="tab"] { color: #626575 !important; background: transparent; }
.stApp [data-testid="stTabs"] [role="tab"] p { color: inherit !important; }
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #45475B !important; border-color: #656A83; }
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] [data-item-hl] {
  background: #fff !important; color: #45475B !important; color-scheme: light;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:is([data-selected="true"],[data-focused="true"]),
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:is([data-selected="true"],[data-focused="true"]) [data-item-hl] {
  background: #EFEDF6 !important; color: #45475B !important;
}


.stApp [data-testid="stButton"] button { min-height: 44px; height: auto; }
.stApp [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
.stApp [data-testid="stButton"] button p {
  white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; -webkit-line-clamp: unset !important;
}
.stApp [data-testid="stTextArea"] textarea {
  border: 1px solid #969AAD !important; border-radius: 10px; padding: 12px;
}
.stApp [data-testid="stIconMaterial"] { color: #45475B !important; }


.stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stExpander"] summary:hover,
.stApp [data-testid="stExpander"] details[open] > summary {
  background: #EAF0F5 !important; color: #45475B !important;
}
.stApp [data-testid="stFormSubmitButton"] button[kind^="primary"]:not(:disabled) :is(p,span) {
  color: #fff !important;
}
.stApp [data-testid="stFormSubmitButton"] button:disabled :is(p,span) { color: #626575 !important; }

/* 소개 영역: 한 줄 머리글로 줄여 답변·근거가 첫 화면에 들어오게 한다. */
.gh-hero { display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap; padding: 22px 28px; border-radius: 18px; margin-bottom: 28px; }
.gh-hero h1 { font-size: clamp(1.3rem, 2.1vw, 1.65rem); line-height: 1.35; margin: 0 0 6px; }
.gh-hero p { font-size: .92rem; }
.gh-hero .gh-eyebrow { margin-bottom: 6px; }
.gh-hero .gh-pill { margin-top: 0; }

/* 상태 요약 카드와 의미 색 배지: 통과(초록)·확인 필요(빨강)·넘김(주황)을 글자와 색으로 함께 구분한다. */
.gh-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 4px 0 22px; }
.gh-stat { background: #fff; border: 1px solid var(--gh-line); border-radius: 14px; padding: 16px 20px; box-shadow: 0 1px 2px rgba(69, 71, 91, .06); }
.gh-stat-label { font-size: .78rem; color: var(--gh-muted); margin-bottom: 8px; }
.gh-stat-value { font-size: 1.2rem; font-weight: 700; color: var(--gh-ink); letter-spacing: -.03em; line-height: 1.4; }
.gh-chip { display: inline-flex; align-items: center; gap: 7px; border-radius: 999px; padding: 3px 12px; font-weight: 700; font-size: .95rem; line-height: 1.6; }
.gh-chip::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.gh-chip.gh-sm { font-size: .74rem; padding: 1px 9px; gap: 5px; vertical-align: 2px; margin-left: 4px; }
.gh-ok { background: #E7F4EC; color: #1F6B3F; }
.gh-bad { background: #FBE9E9; color: #9B2C2C; }
.gh-warn { background: #FFF1D6; color: #855600; }
.gh-tool { display: inline-block; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: .8rem; padding: 3px 10px; border-radius: 8px; margin: 0 6px 4px 0; background: #ECEEF6; color: #3F4566; border: 1px solid #DADDEB; }
.gh-tool-handoff { background: #FFF1D6; color: #855600; border-color: #F0D9A6; }
.gh-tools { margin: 2px 0 8px; }

/* 답변 속 근거 표시 [취업규칙 제13조]를 작은 배지로 */
.gh-cite { display: inline-block; font-size: .74em; font-weight: 600; padding: 0 8px; border-radius: 999px; background: #E9EEF8; color: #3F4F7A; border: 1px solid #D5DDEE; white-space: nowrap; margin: 0 2px; vertical-align: 1px; line-height: 1.7; }

/* 넘김·재작성 안내: 주의(주황)와 안내(파랑)를 나눈다 */
.gh-alert { border: 1px solid; border-radius: 12px; padding: 13px 18px; margin: 0 0 18px; font-size: .95rem; line-height: 1.65; }
.gh-alert strong { font-weight: 700; margin-right: 6px; }
.gh-alert-warn { background: #FFF8E8; border-color: #F0D9A6; color: #6B4800; }
.gh-alert-info { background: #EEF3FA; border-color: #D5DDEE; color: #36456B; }

/* 사이드바 분야 목록: 테두리 상자 대신 가벼운 메뉴 줄 */
.stApp [data-testid="stSidebar"] [data-testid="stExpander"],
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] details { border: none !important; background: transparent !important; box-shadow: none !important; margin-bottom: 0; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary,
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary { background: transparent !important; border-radius: 8px; padding-top: 6px; padding-bottom: 6px; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(101, 106, 131, .09) !important; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-weight: 600; }
.stApp [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding-top: 0; }

/* 선택된 탭 밑줄도 브랜드색으로(테마 설정이 적용되지 않는 버전 대비) */
.stApp [data-testid="stTabs"] [role="tab"][aria-selected="true"] { box-shadow: inset 0 -3px 0 #656A83; }

@media (max-width: 640px) { .gh-stats { grid-template-columns: 1fr; } .gh-hero { padding: 18px 20px; } }

</style>
""", unsafe_allow_html=True)


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
    """조회 도구는 남색, handoff는 주황 배지(HTML)."""
    if not tools:
        return "없음"
    out = []
    for t in tools:
        cls = "gh-tool gh-tool-handoff" if t == context.HANDOFF_TOOL else "gh-tool"
        out.append(f'<span class="{cls}">{escape(t)}</span>')
    return '<div class="gh-tools">' + "".join(out) + "</div>"


def chip(text: str, tone: str, small: bool = False) -> str:
    """tone: ok(초록) | bad(빨강) | warn(주황). 색만이 아니라 글자로도 상태를 적는다."""
    return f'<span class="gh-chip gh-{tone}{" gh-sm" if small else ""}">{escape(text)}</span>'


def alert(kind: str, title: str, body: str) -> str:
    return f'<div class="gh-alert gh-alert-{kind}"><strong>{escape(title)}</strong>{escape(body)}</div>'


def answer_html(text: str) -> str:
    """답변 본문의 [근거 id]를 배지로 바꾼다. 모델 출력은 HTML로 해석되지 않게 먼저 이스케이프한다."""
    safe = md_safe(escape(text or "(답변 없음)", quote=False))
    return re.sub(r"\[([^\[\]\n]{1,80})\]", lambda m: f'<span class="gh-cite">{m.group(1)}</span>', safe)


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
    st.markdown(tool_badges(s.get("tools_called") or []), unsafe_allow_html=True)
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
        st.markdown(alert("warn", "다시 작성함", f"1회차 검증에 걸려 문제 목록을 돌려주고 답변을 다시 썼습니다(총 {len(log)}회 작성)."),
                    unsafe_allow_html=True)
    for v in log:
        mark = chip("통과", "ok", small=True) if v.get("ok") else chip("실패", "bad", small=True)
        st.markdown(f"**{v.get('attempt', '?')}회차** {mark}", unsafe_allow_html=True)
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
    if rec.get("error"):
        st.error(f"처리 중 오류가 났습니다: {rec['error']}")
        return
    s = rec["state"]
    log = s.get("verify_log") or []
    if not log:
        verdict = chip("검증 없음 · 넘김", "warn")
    else:
        verdict = chip("통과", "ok") if log[-1].get("ok") else chip("확인 필요", "bad")
    chunks = evidence_chunks(s)
    used = [cid for cid in cited_ids(s, chunks) if cid in chunks]
    area = escape(category_name(s.get("category")).split(" (")[0])
    st.markdown(f"""<div class="gh-stats">
<div class="gh-stat"><div class="gh-stat-label">담당 분야</div><div class="gh-stat-value">{area}</div></div>
<div class="gh-stat"><div class="gh-stat-label">인용한 근거</div><div class="gh-stat-value">{len(used)}개</div></div>
<div class="gh-stat"><div class="gh-stat-label">기록된 검증 결과</div><div class="gh-stat-value">{verdict}</div></div>
</div>""", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="gh-label">구성원의 질문</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="gh-question">{escape(rec["question"])}</div>', unsafe_allow_html=True)
        kind, note = outcome(s)
        if kind.startswith("handoff"):
            st.markdown(alert("warn", "담당 창구 확인이 필요해요", note), unsafe_allow_html=True)
        elif kind == "partial":
            st.markdown(alert("info", "일부는 담당 창구에 확인해 주세요", note), unsafe_allow_html=True)
        st.markdown('<div class="gh-label">규정에 따른 안내</div>', unsafe_allow_html=True)
        st.markdown(answer_html(s.get("final_answer")), unsafe_allow_html=True)
        st.divider()
        st.markdown("**호출한 도구**")
        st.markdown(tool_badges(s.get("tools_called") or []), unsafe_allow_html=True)
        usage = s.get("usage") or {}
        st.caption(f"기록된 처리 시간 {s.get('seconds', '?')}초 · 입력 {usage.get('input_tokens', 0):,} / 출력 {usage.get('output_tokens', 0):,} 토큰")
        st.write("")
        t1, t2, t3 = st.tabs(["처리 경로", "근거", "검증"])
        with t1:
            show_route(s)
        with t2:
            st.markdown("**근거로 쓴 문서 부분**")
            show_evidence(s)
        with t3:
            st.markdown("**검증 결과**")
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
    st.markdown(f'<div class="gh-brand"><span class="gh-logo">{BRAND_MARK_SVG}</span><div>굿홈앤리빙<small>구성원 안내 데스크</small></div></div>', unsafe_allow_html=True)
    mode = st.radio("이용 방식", ["저장된 실행 기록 보기", "새 질문 보내기"])
    st.divider()
    st.subheader("어떤 내용을 안내하나요?")
    st.caption("근무부터 복리후생까지, 분야에 맞는 사규와 법령을 확인합니다.")
    for key, name, desc in category_guide():
        tool = context.TOOL_OF.get(key, context.HANDOFF_TOOL)
        with st.expander(name):
            st.write(desc)
            st.caption(f"조회 도구 · {tool}")
    st.divider()
    with st.expander("실행 설정"):
        st.caption("현재 새 질문에 적용할 설정입니다. 저장 기록의 설정과는 다를 수 있습니다.")
        st.markdown(f"**분류 모델** `{agent.ROUTER_MODEL}`\n\n**답변 모델** `{agent.ANSWER_MODEL}`\n\n**넘김 기준 확신도** {agent.CONF_THRESHOLD}")
        st.markdown("**개선 스위치**")
        for switch in ("STRICT_SECONDARY", "HANDOFF_STRICT", "ANSWER_RULES_V2", "BOUNDARY_EXAMPLES"):
            st.caption(f"{switch} · {'켜짐' if os.getenv(switch, '0') == '1' else '꺼짐'}")
    if mode == "새 질문 보내기" and st.button("대화 지우기", key="clear", width="stretch"):
        st.session_state.history = []
    st.markdown('<div class="gh-footer">굿홈앤리빙과 사규는 가상 설정입니다.<br>안내는 법률 자문이 아닙니다. 정확한 판단은 경영지원팀 또는 고용노동부 고객상담센터 1350에 확인해 주세요.</div>', unsafe_allow_html=True)

st.markdown("""<div class="gh-hero">
<div>
<div class="gh-eyebrow">굿홈앤리빙 · 구성원 지원</div>
<h1>궁금한 회사 규정, 근거와 함께 확인하세요.</h1>
<p>근무 · 급여 · 복리후생 · 계약 · 고충까지, 답변의 근거와 확인 과정을 한곳에서 살펴보세요.</p>
</div>
<span class="gh-pill">사규와 법령에 기반한 안내</span>
</div>""", unsafe_allow_html=True)

if mode == "저장된 실행 기록 보기":
    st.markdown('<div class="gh-section">저장된 실행 기록 보기</div>', unsafe_allow_html=True)
    st.caption("평가 기록 재생 · 새 답변을 생성하지 않으며 모델을 호출하지 않습니다.")
    path = ROOT / "runs" / "05_strict_secondary_r1" / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = st.selectbox("살펴볼 질문을 선택하세요", range(len(rows)), format_func=lambda i: f"{rows[i]['id']} · {rows[i].get('question', '오류 기록')}")
    st.caption("기록 설정: STRICT_SECONDARY=1 · 검증은 당시 결과, 근거 원문은 현재 문서를 표시합니다.")
    st.write("")
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
    for ex in examples:
        st.button(ex["question"], key=f"ex_{ex['id']}", width="stretch",
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
