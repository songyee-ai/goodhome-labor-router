"""평가셋으로 파이프라인을 측정한다.

지표
1. 분류: 정확도, macro F1, 혼동 행렬 (category가 있는 문항, 분류기 출력 기준)
2. 도구 호출 적절성: 실제로 호출한 도구 집합 == expected_tools 이면 1점
3. 답변 적절성: must_include를 모두 담고, must_not_include를 하나도 어기지 않고,
   forbidden_numbers가 하나도 나오지 않으면 1점 (사실 판단은 채점 모델, 금지 숫자는 코드)

사용
  python evaluate.py run --label 03_x --repeat 3          # 평가셋 전체를 3번 실행 → runs/03_x_r1~r3/
  python evaluate.py regrade --src a,b --label new        # 저장된 답변을 지금 정답지로 다시 채점
  python evaluate.py compare --labels a,b,c               # 여러 실행의 평균·범위
  python evaluate.py judge-check                          # 채점기 검증(data/judge_validation.json)
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()  # JUDGE_MODEL·EVAL_WORKERS를 읽기 전에 .env를 먼저 불러온다

ROOT = Path(__file__).resolve().parent
GOLDENSET = ROOT / "data" / "goldenset.json"
JUDGE_SET = ROOT / "data" / "judge_validation.json"
RUNS = ROOT / "runs"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.6-sol")
LABELS = ["work", "pay", "welfare", "contract", "harassment", "out_of_scope"]
WORKERS = int(os.getenv("EVAL_WORKERS", "6"))


def mask_secrets(text: str) -> str:
    """오류 문구에 API 키('sk-…' 꼴)가 섞여 있으면 가린다. 결과 파일·보고서에 키가 남지 않게."""
    return re.sub(r"sk-[A-Za-z0-9_*-]{6,}", "sk-***", text)


# ── 채점기 ────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
당신은 사내 규정 안내 답변을 채점합니다. 표현이 아니라 사실을 봅니다.

- 필수 사실: 답변이 그 사실을 같은 뜻으로 전달하면 포함입니다. 문장·어순·단위 표기(예: 6일/엿새, 187,000원/18만 7천 원)가 달라도 됩니다.
  숫자·대상·조건이 다르면 포함이 아닙니다. 행동(예: "담당 창구를 안내한다")은 답변이 실제로 그 행동을 했으면 포함입니다.
- 금지 내용: 답변이 그 내용을 사실로 주장하거나 권하면 위반입니다. 답변이 그 내용을 "아니다", "그렇지 않다"고 부정하거나,
  법 기준과 사규 기준을 구분해 설명하면서 언급만 한 것은 위반이 아닙니다.
- 항목마다 주어진 번호(I1, I2 … 는 필수 사실, N1, N2 … 는 금지 내용)로 하나씩 판단합니다. 문장을 다시 옮겨 적지 않습니다.
- ok는 필수 사실이면 "포함됨", 금지 내용이면 "위반하지 않음"일 때 true입니다.
- 판단마다 답변에서 근거가 된 구절을 짧게 인용합니다. 없으면 빈 문자열."""


class FactCheck(BaseModel):
    id: str = Field(description="항목 번호. 예: I1, N2")
    ok: bool
    quote: str


class JudgeResult(BaseModel):
    checks: list[FactCheck]


def judge(question: str, answer: str, must_include: list[str], must_not: list[str]) -> dict:
    """항목 번호로 판단을 받아 정답지 문장에 다시 붙인다. 채점 모델이 문장을 옮겨 적다 섞는 일을 막는다."""
    from agent import client  # 같은 OpenAI 클라이언트를 쓴다

    items = {f"I{k + 1}": t for k, t in enumerate(must_include)}
    items |= {f"N{k + 1}": t for k, t in enumerate(must_not)}
    user = json.dumps({"질문": question, "답변": answer,
                       "필수 사실": {k: v for k, v in items.items() if k[0] == "I"},
                       "금지 내용": {k: v for k, v in items.items() if k[0] == "N"}},
                      ensure_ascii=False, indent=1)
    got = {}
    for _ in range(2):
        resp = client().responses.parse(
            model=JUDGE_MODEL,
            input=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            text_format=JudgeResult,
        )
        new = {c.id.strip(): c for c in resp.output_parsed.checks if c.id.strip() in items}
        got |= {k: v for k, v in new.items() if k not in got}  # 두 결과를 합친다. 먼저 받은 판단이 우선
        if len(got) == len(items):
            break
    lost = [k for k in items if k not in got]  # 두 번 물어도 판단이 없는 항목은 안전하게 불합격 처리
    return {
        "must_include": [{"id": k, "item": items[k], "ok": got[k].ok if k in got else False,
                          "quote": got[k].quote if k in got else ""} for k in items if k[0] == "I"],
        "must_not_include": [{"id": k, "item": items[k], "ok": got[k].ok if k in got else False,
                              "quote": got[k].quote if k in got else ""} for k in items if k[0] == "N"],
        "judge_error": lost,
    }


def forbidden_hits(answer: str, forbidden: list[str]) -> list[str]:
    """금지 숫자가 답변에 나오는지(공백은 무시).

    더 큰 숫자의 일부는 일치로 보지 않는다. '5일'은 '15일'·'1.5일'에, '200,000원'은 '1,200,000원'에 안 걸린다.
    앞쪽은 숫자가 바로 붙어 있거나 '숫자+쉼표/마침표'가 붙어 있을 때만 막는다. 문장부호 뒤에 이어진
    '있습니다. 5일'·'6일, 5일'은 따로 센 숫자이므로 걸린다. 금지 문자열이 숫자로 끝나면 뒤에 이어지는
    숫자('50'에 대한 '500'·'50.5'·'50,000')도 같은 이유로 막는다.
    """
    text = answer.replace(" ", "")
    hits = []
    for f in forbidden:
        f2 = f.replace(" ", "")
        if not f2:
            continue
        tail = r"(?![\d,.]?\d)" if f2[-1].isdigit() else ""
        if re.search(rf"(?<!\d)(?<!\d[,.]){re.escape(f2)}{tail}", text):
            hits.append(f)
    return hits


def grade_answer(item: dict, answer: str) -> dict:
    j = judge(item["question"], answer, item["must_include"], item["must_not_include"])
    missing = [f["item"] for f in j["must_include"] if not f["ok"]]
    violated = [f["item"] for f in j["must_not_include"] if not f["ok"]]
    hits = forbidden_hits(answer, item["forbidden_numbers"])
    return {
        "score": int(not missing and not violated and not hits),
        "missing": missing,
        "violated": violated,
        "forbidden_hits": hits,
        "judge_error": j["judge_error"],
        "judge": j,
    }


# ── 분류 지표 ─────────────────────────────────────────────────────────

def classification_metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (정답, 예측). 정답이 있는 문항만.

    macro F1은 정답이나 예측에 한 번이라도 나온 칸만 평균한다(한 번도 안 나온 칸을 0점으로 세지 않는다).
    평균에 쓴 칸은 결과의 macro_labels에 담는다.
    """
    matrix = {g: Counter() for g in LABELS}
    for g, p in pairs:
        matrix[g][p] += 1
    per = {}
    for c in LABELS:
        tp = matrix[c][c]
        fp = sum(matrix[g][c] for g in LABELS if g != c)
        fn = sum(n for p, n in matrix[c].items() if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(matrix[c].values())}
    seen = {g for g, _ in pairs} | {p for _, p in pairs}
    used = [c for c in LABELS if c in seen]
    acc = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0
    macro = sum(per[c]["f1"] for c in used) / len(used) if used else 0.0
    return {"accuracy": acc, "macro_f1": macro, "macro_labels": used, "per_class": per,
            "confusion": {g: dict(matrix[g]) for g in LABELS}}


def confusion_table(conf: dict) -> str:
    head = "| 정답 \\ 예측 | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS)
    rows = [f"| {g} | " + " | ".join(str(conf[g].get(p, 0) or "·") for p in LABELS) + " |" for g in LABELS]
    return head + "\n" + "\n".join(rows)


# ── 실행 ──────────────────────────────────────────────────────────────

def evaluate_item(item: dict) -> dict:
    import agent

    try:
        s = agent.run(item["question"])
    except Exception as e:  # 한 문항 실패가 전체 측정을 멈추지 않게
        return {"id": item["id"], "error": mask_secrets(repr(e))}
    tools = sorted(set(s.get("tools_called", [])))
    rec = {
        "id": item["id"],
        "question": item["question"],
        "gold_category": item["category"],
        "gold_secondary": item["secondary"],
        "pred_category": s.get("category"),
        "pred_secondary": s.get("secondary"),
        "confidence": s.get("confidence"),
        "route": s.get("route"),
        "expected_tools": sorted(item["expected_tools"]),
        "tools_called": tools,
        "tool_score": int(tools == sorted(item["expected_tools"])),
        "answer": s.get("final_answer", ""),
        "citations": s.get("citations"),
        "uncovered": s.get("uncovered"),
        "verify_log": s.get("verify_log", []),
        "seconds": s.get("seconds"),
        "usage": s.get("usage", {}),
    }
    try:
        rec.update(grade_answer(item, rec["answer"]))
    except Exception as e:  # 채점 실패는 오류만 붙이고 답변 기록은 남긴다(regrade로 다시 채점할 수 있게)
        rec["error"] = f"채점 실패: {mask_secrets(repr(e))}"
    return rec


def summarize(results: list[dict], items: dict) -> dict:
    ok = [r for r in results if "error" not in r]
    if not ok:  # 모든 문항이 실패(예: API 한도 초과)해도 오류 내용은 저장되게 한다
        first = results[0]["error"] if results else "문항 없음"
        raise SystemExit(f"모든 문항이 실패해 지표를 계산할 수 없습니다. 첫 오류: {first}")
    pairs = [(r["gold_category"], r["pred_category"]) for r in ok if r["gold_category"] is not None]
    cls = classification_metrics(pairs)
    unclear = [r for r in ok if r["gold_category"] is None]
    by_type = defaultdict(lambda: [0, 0, 0])
    for r in ok:
        t = by_type[items[r["id"]]["type"]]
        t[0] += 1
        t[1] += r["tool_score"]
        t[2] += r["score"]
    retried = sum(1 for r in ok if len(r["verify_log"]) > 1)
    blocked = sum(1 for r in ok if r["verify_log"] and not r["verify_log"][-1]["ok"])
    return {
        "n": len(results),
        "errors": [r["id"] for r in results if "error" in r],
        "classification": cls,
        "unclear_handed_off": f"{sum(1 for r in unclear if r['route'] != 'answer')}/{len(unclear)}",
        "n_errors": len(results) - len(ok),
        "tool_accuracy": sum(r["tool_score"] for r in ok) / len(ok),
        "answer_accuracy": sum(r["score"] for r in ok) / len(ok),
        # 위 두 지표는 오류 문항을 분모에서 뺀다(기존 실행과 비교하려고 유지). 아래는 오류를 0점으로 센 전체 기준
        "tool_accuracy_all": sum(r["tool_score"] for r in ok) / len(results),
        "answer_accuracy_all": sum(r["score"] for r in ok) / len(results),
        "by_type": {k: {"n": v[0], "tool": v[1], "answer": v[2]} for k, v in by_type.items()},
        "verify_retried": retried,
        "verify_blocked": blocked,
        "avg_seconds": sum(r["seconds"] or 0 for r in ok) / len(ok),
        "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in ok),
        "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in ok),
    }


def write_report(out: Path, label: str, summ: dict, results: list[dict], note: str):
    c = summ["classification"]
    lines = [
        f"# 측정 결과: {label}",
        "",
        f"- 일시: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 메모: {note or '-'}",
        f"- 모델: 분류 {os.getenv('ROUTER_MODEL', 'gpt-5.6-luna')}, 답변 {os.getenv('ANSWER_MODEL', 'gpt-5.6-terra')}, 채점 {JUDGE_MODEL}",
        "- 개선 스위치: " + (", ".join(f"{k}={os.getenv(k)}" for k in
                                     ("ANSWER_RULES_V2", "STRICT_SECONDARY", "BOUNDARY_EXAMPLES", "HANDOFF_STRICT")
                                     if os.getenv(k) == "1")
                           or "모두 꺼짐"),
        f"- 문항 {summ['n']}건, 오류 {len(summ['errors'])}건 {summ['errors'] or ''}",
        "",
        "## 요약",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 분류 정확도 | {c['accuracy']:.3f} |",
        f"| 분류 macro F1 | {c['macro_f1']:.3f} |",
        f"| 모호 문항 넘김 | {summ['unclear_handed_off']} |",
        f"| 도구 호출 적절성 | {summ['tool_accuracy']:.3f} |",
        f"| 답변 적절성 | {summ['answer_accuracy']:.3f} |",
        f"| 도구 호출 적절성 (전체 기준, 오류 0점) | {summ['tool_accuracy_all']:.3f} |",
        f"| 답변 적절성 (전체 기준, 오류 0점) | {summ['answer_accuracy_all']:.3f} |",
        f"| 오류 문항 | {summ['n_errors']} |",
        f"| 검증 재작성 / 차단 | {summ['verify_retried']} / {summ['verify_blocked']} |",
        f"| 평균 소요 | {summ['avg_seconds']:.1f}초 |",
        f"| 토큰(입력/출력) | {summ['input_tokens']:,} / {summ['output_tokens']:,} |",
        "",
        "## 혼동 행렬 (분류기 출력 기준)",
        "",
        confusion_table(c["confusion"]),
        "",
        "## 칸별 성능",
        "",
        "| 칸 | 정밀도 | 재현율 | F1 | 문항 |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {k} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |"
              for k, v in c["per_class"].items()]
    lines += ["", f"macro F1은 정답이나 예측에 한 번이라도 나온 칸만 평균한 값: {', '.join(c['macro_labels']) or '-'} "
                  f"({len(c['macro_labels'])}칸)",
              "도구·답변 적절성은 오류 문항을 뺀 값이고, 오류를 0점으로 센 전체 기준 값은 따로 적었다."]
    lines += ["", "## 유형별", "", "| 유형 | 문항 | 도구 맞음 | 답변 맞음 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['tool']} | {v['answer']} |" for k, v in sorted(summ["by_type"].items())]
    lines += ["", "## 틀린 문항", ""]
    for r in results:
        if "error" in r:
            kept = " (답변은 저장돼 있어 regrade로 다시 채점할 수 있음)" if r.get("answer") else ""
            lines.append(f"- **{r['id']}** 오류: {r['error']}{kept}")
            continue
        if r["tool_score"] and r["score"]:
            continue
        lines.append(f"### {r['id']} — 도구 {'O' if r['tool_score'] else 'X'} / 답변 {'O' if r['score'] else 'X'}")
        lines.append(f"- 질문: {r['question']}")
        lines.append(f"- 분류: 정답 {r['gold_category']}/{r['gold_secondary']} → 예측 {r['pred_category']}/{r['pred_secondary']} "
                     f"(확신도 {r['confidence']}, 경로 {r['route']})")
        lines.append(f"- 도구: 기대 {r['expected_tools']} / 실제 {r['tools_called']}")
        if r["missing"]:
            lines.append(f"- 빠진 사실: {r['missing']}")
        if r["violated"]:
            lines.append(f"- 어긴 금지 내용: {r['violated']}")
        if r["forbidden_hits"]:
            lines.append(f"- 나온 금지 숫자: {r['forbidden_hits']}")
        if r.get("judge_error"):
            lines.append(f"- 채점 오류(판단을 못 받은 항목): {r['judge_error']}")
        if r["verify_log"]:
            lines.append(f"- 검증: {[('통과' if v['ok'] else v['problems']) for v in r['verify_log']]}")
        lines.append(f"- 답변: {r['answer']}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


HISTORY_COLUMNS = ["일시", "label", "메모", "문항", "분류 정확도", "macro F1", "도구 호출", "답변", "재작성", "차단",
                   "오류", "도구 호출(전체)", "답변(전체)"]  # 뒤의 세 열은 나중에 추가된 것이라 맨 뒤에 둔다


def append_history(hist: Path, row: dict):
    """history.csv에 한 줄 더한다. 기존 파일의 머리글에 없는 열은 머리글 끝에만 덧붙이고 기존 행은 그대로 둔다."""
    if not hist.exists():
        header = HISTORY_COLUMNS
        with hist.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(header)
    else:
        with hist.open(encoding="utf-8-sig", newline="") as f:
            text = f.read()
        first, sep, rest = re.match(r"([^\r\n]*)(\r?\n?)(.*)", text, re.S).groups()
        header = next(csv.reader([first]))
        added = [c for c in HISTORY_COLUMNS if c not in header]
        if added:  # 새 행의 열 수가 머리글과 같아지도록 머리글 줄만 고쳐 쓴다
            header += added
            buf = io.StringIO()
            csv.writer(buf).writerow(header)
            with hist.open("w", encoding="utf-8-sig", newline="") as f:
                f.write(buf.getvalue().rstrip("\r\n") + (sep or "\r\n") + rest)
    with hist.open("a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in header])


def save_run(label: str, note: str, results: list[dict], items: list[dict], wall: float):
    out = RUNS / label
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:  # 지표 계산 전에 먼저 남긴다
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarize(results, {i["id"]: i for i in items})
    summ["wall_seconds"] = round(wall, 1)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out, label, summ, results, note)

    c = summ["classification"]
    append_history(RUNS / "history.csv", {
        "일시": f"{datetime.now():%Y-%m-%d %H:%M}", "label": label, "메모": note, "문항": summ["n"],
        "분류 정확도": f"{c['accuracy']:.3f}", "macro F1": f"{c['macro_f1']:.3f}",
        "도구 호출": f"{summ['tool_accuracy']:.3f}", "답변": f"{summ['answer_accuracy']:.3f}",
        "재작성": summ["verify_retried"], "차단": summ["verify_blocked"],
        "오류": summ["n_errors"], "도구 호출(전체)": f"{summ['tool_accuracy_all']:.3f}",
        "답변(전체)": f"{summ['answer_accuracy_all']:.3f}"})
    print(f"[{label}] 분류 정확도 {c['accuracy']:.3f} · macro F1 {c['macro_f1']:.3f} · "
          f"도구 {summ['tool_accuracy']:.3f} · 답변 {summ['answer_accuracy']:.3f} · "
          f"오류 {summ['n_errors']}건(전체 기준 도구 {summ['tool_accuracy_all']:.3f} · 답변 {summ['answer_accuracy_all']:.3f}) · "
          f"{summ['wall_seconds']}초 → {out / 'report.md'}")


def cmd_run(args):
    items = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    if args.only:
        items = [i for i in items if i["id"] in args.only.split(",")]
    for k in range(1, args.repeat + 1):
        label = args.label if args.repeat == 1 else f"{args.label}_r{k}"
        start = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(evaluate_item, items))
        save_run(label, args.note, results, items, time.time() - start)
    if args.repeat > 1:
        print_compare([f"{args.label}_r{k}" for k in range(1, args.repeat + 1)])


def cmd_regrade(args):
    """에이전트를 다시 돌리지 않고, 저장된 답변·도구 기록을 지금의 정답지로 다시 채점한다(정답지 변경 효과만 보기)."""
    items = json.loads(GOLDENSET.read_text(encoding="utf-8"))
    gold = {i["id"]: i for i in items}
    for src in args.src.split(","):
        old = [json.loads(l) for l in (RUNS / src / "results.jsonl").open(encoding="utf-8")]
        skipped = [r["id"] for r in old if r["id"] not in gold]  # 정답지에서 빠진 문항은 채점할 수 없다
        old = [r for r in old if r["id"] in gold]
        changed = [r["id"] for r in old if r.get("question", gold[r["id"]]["question"]) != gold[r["id"]]["question"]]
        if skipped:
            print(f"[경고] {src}: 지금 정답지에 없는 문항 {skipped}은 건너뜀")
        if changed:
            print(f"[경고] {src}: 질문이 정답지와 달라진 문항 {changed} (옛 질문에 한 답변을 새 질문 기준으로 채점함)")

        def one(r):
            if "answer" not in r:  # 에이전트 실행 자체가 실패한 기록은 다시 채점할 것이 없다
                return r
            item = gold[r["id"]]
            new = {k: v for k, v in r.items() if k != "error"}  # 이전 채점 실패 표시는 새로 채점해 보고 정한다
            new.update(gold_category=item["category"], gold_secondary=item["secondary"],
                       expected_tools=sorted(item["expected_tools"]))
            new["tool_score"] = int(new["tools_called"] == new["expected_tools"])
            try:
                new.update(grade_answer(item, new["answer"]))
            except Exception as e:  # 채점 실패는 그 문항만 오류로 남기고(답변 등 기록은 보존) 계속한다
                new["error"] = f"채점 실패: {mask_secrets(repr(e))}"
            return new

        start = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            results = list(pool.map(one, old))
        label = f"{args.label}{src[len(args.strip):] if args.strip and src.startswith(args.strip) else '_' + src}"
        note = f"{args.note} (답변은 {src} 그대로, 다시 채점)"
        if skipped:
            note += f" 정답지에 없어 건너뜀: {','.join(skipped)}"
        if changed:
            note += f" 질문이 달라진 문항: {','.join(changed)}"
        save_run(label, note, results, items, time.time() - start)


def print_compare(labels: list[str]):
    rows = []
    for lb in labels:
        s = json.loads((RUNS / lb / "summary.json").read_text(encoding="utf-8"))
        n_err = s.get("n_errors", len(s.get("errors", [])))  # 오류 수를 안 적던 옛 실행도 읽는다
        ok_ratio = (s["n"] - n_err) / s["n"] if s["n"] else 0.0
        rows.append((s["classification"]["accuracy"], s["classification"]["macro_f1"],
                     s["tool_accuracy"], s["answer_accuracy"], s["verify_blocked"], n_err,
                     s.get("tool_accuracy_all", s["tool_accuracy"] * ok_ratio),
                     s.get("answer_accuracy_all", s["answer_accuracy"] * ok_ratio)))
    names = ["분류 정확도", "macro F1", "도구 호출", "답변", "차단", "오류", "도구 호출(전체)", "답변(전체)"]
    parts = []
    for k, n in enumerate(names):
        fmt = ".1f" if n in ("차단", "오류") else ".3f"  # 건수는 소수 한 자리, 비율은 세 자리
        vals = [r[k] for r in rows]
        parts.append(f"{n} 평균 {sum(vals) / len(vals):{fmt}} (범위 {min(vals):{fmt}}~{max(vals):{fmt}})")
    print(f"\n{len(labels)}회 요약: " + " · ".join(parts))


def cmd_compare(args):
    print_compare(args.labels.split(","))


def cmd_judge_check(args):
    """채점기 검증: 사람이 정답을 아는 답안(통과해야 할 모범 답안, 떨어져야 할 오답)을 채점기에 넣어 본다."""
    cases = json.loads(JUDGE_SET.read_text(encoding="utf-8"))
    gold = {i["id"]: i for i in json.loads(GOLDENSET.read_text(encoding="utf-8"))}

    def one(case):
        try:
            g = grade_answer(gold[case["id"]], case["answer"])
        except Exception as e:  # 한 문항의 채점 실패가 검증 전체를 멈추지 않게. 실패한 답안으로 따로 센다
            return {**case, "got": None, "missing": [], "violated": [], "forbidden_hits": [],
                    "judge_error": [], "error": f"채점 실패: {mask_secrets(repr(e))}"}
        return {**case, "got": g["score"], "missing": g["missing"], "violated": g["violated"],
                "forbidden_hits": g["forbidden_hits"], "judge_error": g["judge_error"]}

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, cases))

    def right_reason(r):
        """탈락이 맞게 나왔을 때, 일부러 넣은 결함을 이유로 탈락시켰는가."""
        if r["kind"] == "missing_fact":
            return r["mutation"] in r["missing"] or any(r["mutation"][:15] in m for m in r["missing"])
        if r["kind"] == "violation":
            return bool(r["violated"] or r["forbidden_hits"])
        return True

    kinds = defaultdict(lambda: [0, 0, 0])
    wrong = []
    for r in rows:
        k = kinds[r["kind"]]
        k[0] += 1
        k[1] += int(r["got"] == r["expect"])
        k[2] += int(r["got"] == r["expect"] and right_reason(r))
        if "error" not in r and (r["got"] != r["expect"] or not right_reason(r)):
            wrong.append(r)
    failed = [r for r in rows if "error" in r]  # 채점 중 예외가 난 답안(판정 불일치로 세지 않고 따로 적는다)
    total = sum(v[1] for v in kinds.values())
    errs = sum(1 for r in rows if r["judge_error"])
    lines = ["# 채점기 검증", "", f"- 일시: {datetime.now():%Y-%m-%d %H:%M}, 채점 모델 {JUDGE_MODEL}",
             f"- 전체 일치: {total}/{len(rows)}, 채점 오류(항목 누락): {errs}건, 채점 실패(예외): {len(failed)}건", "",
             "| 답안 종류 | 기대 | 건수 | 판정 일치 | 이유까지 일치 |", "|---|---|---|---|---|"]
    expect_of = {r["kind"]: r["expect"] for r in rows}
    lines += [f"| {k} | {'통과' if expect_of[k] else '탈락'} | {v[0]} | {v[1]} | {v[2]} |" for k, v in kinds.items()]
    lines += ["", "이유까지 일치: 사실 빠진 답안은 뺀 그 사실을 '빠짐'으로 짚었는지, "
              "금지 내용 섞인 답안은 '금지 위반'(또는 금지 숫자)으로 짚었는지 본다."]
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "judge_check.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines += ["", "## 판정이나 이유가 어긋난 답안", ""]
    for r in wrong:
        lines += [f"### {r['id']} ({r['kind']}) 기대 {r['expect']} → 채점 {r['got']}",
                  f"- 넣은 결함: {r['mutation'] or '-'}",
                  f"- 답안: {r['answer']}", f"- 빠졌다고 본 사실: {r['missing']}",
                  f"- 어겼다고 본 금지 내용: {r['violated']}", f"- 금지 숫자: {r['forbidden_hits']}", ""]
    if failed:
        lines += ["## 채점이 실패한 답안", ""]
        lines += [f"- {r['id']} ({r['kind']}): {r['error']}" for r in failed] + [""]
    out = RUNS / "judge_check.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"채점기 일치 {total}/{len(rows)}, 채점 오류 {errs}건, 채점 실패 {len(failed)}건 → {out}")
    for k, v in kinds.items():
        print(f"  {k}: 판정 {v[1]}/{v[0]}, 이유까지 {v[2]}/{v[0]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--only", default="", help="쉼표로 구분한 문항 id만 실행")
    r.add_argument("--repeat", type=int, default=1, help="같은 설정을 여러 번 돌려 변동 폭을 본다")
    g = sub.add_parser("regrade")
    g.add_argument("--src", required=True, help="다시 채점할 실행 이름(쉼표로 여러 개)")
    g.add_argument("--label", required=True)
    g.add_argument("--strip", default="", help="src 이름에서 이 앞부분을 떼고 나머지를 label 뒤에 붙인다")
    g.add_argument("--note", default="")
    c = sub.add_parser("compare")
    c.add_argument("--labels", required=True)
    sub.add_parser("judge-check")
    args = p.parse_args()
    {"run": cmd_run, "regrade": cmd_regrade, "compare": cmd_compare,
     "judge-check": cmd_judge_check}[args.cmd](args)


if __name__ == "__main__":
    main()
