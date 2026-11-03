"""evaluate.py의 코드 부분(금지 숫자 검사, 분류 지표, 기록·재채점 처리) 시험과 채점 모델 연결 확인.

실행: .venv\\Scripts\\python.exe tests/test_evaluate.py [--judge]
  인자 없이 돌리면 모델을 부르지 않는다(채점 모델·에이전트는 가짜로 바꿔 끼운다).
  --judge 를 붙이면 예시 문항 하나로 채점 모델을 실제로 한 번 부른다.
"""
import csv
import io
import json
import shutil
import sys
import tempfile
import types
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402
from evaluate import classification_metrics, forbidden_hits, grade_answer  # noqa: E402

fails = 0


def check(ok: bool, msg: str):
    global fails
    fails += not ok
    print(f"[{'OK ' if ok else '틀림'}] {msg}")


@contextmanager
def patched(obj, **attrs):
    """모듈 속성을 잠깐 바꿔 끼운다."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextmanager
def captured():
    """표준출력을 가로챈다. 결과는 yield한 함수로 읽는다."""
    raw = io.BytesIO()
    wrapper = io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
    old, sys.stdout = sys.stdout, wrapper
    try:
        yield lambda: (wrapper.flush(), raw.getvalue().decode("utf-8"))[1]  # wrapper를 붙들어 raw가 닫히지 않게 한다
    finally:
        sys.stdout = old


@contextmanager
def fake_agent(run=None, replies=None):
    """agent 모듈을 가짜로 바꿔 끼운다. 모델을 부르지 않는다.

    run: agent.run 대신 쓸 함수. replies: 채점 모델이 차례로 돌려줄 [(id, ok), …] 목록의 목록.
    """
    calls = iter(replies or [])

    class Resp:
        def parse(self, **kw):
            checks = [types.SimpleNamespace(id=i, ok=ok, quote="") for i, ok in next(calls)]
            return types.SimpleNamespace(output_parsed=types.SimpleNamespace(checks=checks))

    mod = types.ModuleType("agent")
    mod.run = run
    mod.client = lambda: types.SimpleNamespace(responses=Resp())
    old = sys.modules.get("agent")
    sys.modules["agent"] = mod
    try:
        yield
    finally:
        if old is None:
            del sys.modules["agent"]
        else:
            sys.modules["agent"] = old


def item(iid, question="질문", **kw):
    base = {"id": iid, "question": question, "category": "work", "secondary": None, "type": "normal",
            "expected_tools": ["lookup_work"], "must_include": ["사실"], "must_not_include": [],
            "forbidden_numbers": []}
    return {**base, **kw}


def graded_ok(question, answer, must_include, must_not):
    """가짜 채점기: 모든 항목을 통과로 본다. 'BOOM'이 답변에 있으면 채점이 예외로 실패한다."""
    if "BOOM" in answer:
        raise RuntimeError("채점 실패 흉내 " + "sk-" + "proj-abcdef123456")
    return {"must_include": [{"id": f"I{k + 1}", "item": t, "ok": True, "quote": ""} for k, t in enumerate(must_include)],
            "must_not_include": [{"id": f"N{k + 1}", "item": t, "ok": True, "quote": ""} for k, t in enumerate(must_not)],
            "judge_error": []}


def rec(iid, answer="답", **kw):
    base = {"id": iid, "question": "질문", "gold_category": "work", "gold_secondary": None,
            "pred_category": "work", "pred_secondary": None, "confidence": 0.9, "route": "answer",
            "expected_tools": ["lookup_work"], "tools_called": ["lookup_work"], "tool_score": 1,
            "answer": answer, "citations": [], "uncovered": [], "verify_log": [], "seconds": 1.0,
            "usage": {"input_tokens": 10, "output_tokens": 5}, "score": 1, "missing": [],
            "violated": [], "forbidden_hits": [], "judge_error": []}
    return {**base, **kw}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# ── 시험 묶음 ─────────────────────────────────────────────────────────

def test_forbidden_numbers():
    cases = [
        ("결혼 휴가는 6일입니다", ["5일"], []),
        ("가족돌봄휴가는 15일까지", ["5일"], []),          # 앞에 숫자가 붙으면 일치 아님
        ("식대는 200,000원입니다", ["200,000원"], ["200,000원"]),
        ("식대는 1,200,000원입니다", ["200,000원"], []),    # 숫자에 붙은 쉼표면 일치 아님
        ("야간 50 % 가산", ["50%"], ["50%"]),              # 공백 무시
        ("결혼 휴가가 있습니다. 5일입니다.", ["5일"], ["5일"]),  # 문장 끝 마침표 뒤의 5일은 따로 센 숫자
        ("본인 6일, 5일은 자녀", ["5일"], ["5일"]),         # 쉼표 뒤의 5일도 마찬가지
        ("본인 6일,5일은 자녀", ["5일"], ["5일"]),
        ("연차는 1.5일입니다", ["5일"], []),                # 소수점에 붙은 5일은 일치 아님
        ("식대는 1,200,000원과 200,000원", ["200,000원"], ["200,000원"]),  # 뒤쪽에 진짜 200,000원이 있으면 걸림
        ("가산율은 500%입니다", ["50"], []),                # 숫자로 끝나는 금지 문자열: 뒤에 숫자가 이어지면 아님
        ("가산율은 50.5입니다", ["50"], []),
        ("가산율은 50,000입니다", ["50"], []),
        ("가산율은 50입니다.", ["50"], ["50"]),             # 문장 끝 마침표는 이어진 숫자가 아님
        ("가산율은 50%입니다", ["50"], ["50"]),
        ("연차 3일", [""], []),                            # 빈 금지 문자열은 무시
    ]
    for text, forb, want in cases:
        got = forbidden_hits(text, forb)
        check(got == want, f"금지 숫자 {text!r} {forb} → {got}")


def test_classification():
    m = classification_metrics([("work", "work"), ("work", "pay"), ("pay", "pay"), ("welfare", "welfare")])
    check(abs(m["accuracy"] - 0.75) < 1e-9 and m["confusion"]["work"]["pay"] == 1,
          f"분류 지표 정확도 {m['accuracy']:.2f}, work→pay {m['confusion']['work'].get('pay')}")
    m = classification_metrics([("work", "work"), ("pay", "pay")])
    check(m["macro_labels"] == ["work", "pay"] and abs(m["macro_f1"] - 1.0) < 1e-9,
          f"macro F1은 나온 칸만 평균: {m['macro_labels']} → {m['macro_f1']:.2f}")
    m = classification_metrics([("work", "work"), ("work", "out_of_scope")])
    check(m["macro_labels"] == ["work", "out_of_scope"] and abs(m["macro_f1"] - (2 / 3) / 2) < 1e-9,
          f"예측에만 나온 칸도 평균에 넣음: {m['macro_labels']} → {m['macro_f1']:.3f}")
    m = classification_metrics([])
    check(m["accuracy"] == 0.0 and m["macro_f1"] == 0.0 and m["macro_labels"] == [], "정답 있는 문항이 없어도 0으로 나누지 않음")


def test_mask_secrets():
    s = evaluate.mask_secrets("Incorrect API key: " + "sk-" + "proj-AbC_123-xyz and " + "sk-" + "abcdef123456 끝")
    check("sk-proj" not in s and "abcdef" not in s and s.count("sk-***") == 2, f"비밀값 가림 → {s}")
    check(evaluate.mask_secrets("task-1 sk-1") == "task-1 sk-1", "sk- 꼴이 아니면 그대로 둠")


def test_judge_merge():
    mi, mn = ["가", "나"], ["다"]
    # 첫 답에 I1만, 둘째 답에 I1(다른 판단)·I2·N1: 첫 판단(I1 통과)은 유지되고 나머지가 채워진다
    with fake_agent(replies=[[("I1", True)], [("I1", False), ("I2", True), ("N1", True)]]):
        j = evaluate.judge("질문", "답", mi, mn)
    oks = {f["id"]: f["ok"] for f in j["must_include"] + j["must_not_include"]}
    check(oks == {"I1": True, "I2": True, "N1": True} and j["judge_error"] == [],
          f"재질문 결과를 합치고 첫 결과를 우선 {oks}")
    with fake_agent(replies=[[("I1", True)], [("N1", True)]]):
        j = evaluate.judge("질문", "답", mi, mn)
    check(j["judge_error"] == ["I2"] and j["must_include"][0]["ok"] is True and j["must_not_include"][0]["ok"] is True,
          f"끝내 판단이 없는 항목만 불합격 처리 {j['judge_error']}")


def test_evaluate_item():
    it = item("W01", question="휴가?")
    with patched(evaluate, judge=graded_ok), fake_agent(
            run=lambda q: {"category": "work", "tools_called": ["lookup_work"], "final_answer": "BOOM"}):
        r = evaluate.evaluate_item(it)
    check("error" in r and r["answer"] == "BOOM" and r["tool_score"] == 1 and r["id"] == "W01",
          "채점이 실패해도 답변·도구 기록은 남기고 error만 붙임")
    check("sk-proj" not in r["error"] and "sk-***" in r["error"], f"오류 문구의 키는 가려서 기록 {r['error']}")

    def boom(q):
        raise RuntimeError("실행 실패 sk-abcdef123456")

    with fake_agent(run=boom):
        r = evaluate.evaluate_item(it)
    check(set(r) == {"id", "error"} and "abcdef" not in r["error"], f"에이전트 실행 실패는 id·error만 {r}")
    with patched(evaluate, judge=graded_ok), fake_agent(
            run=lambda q: {"category": "work", "tools_called": ["lookup_work"], "final_answer": "정상"}):
        r = evaluate.evaluate_item(it)
    check("error" not in r and r["score"] == 1, "정상 문항에는 error 키가 없음")


def test_summarize_and_report(tmp: Path):
    items = {"W01": {"type": "normal"}, "W02": {"type": "normal"}, "W03": {"type": "normal"}}
    results = [rec("W01"), rec("W02", tool_score=0, score=0),
               {**rec("W03"), "error": "채점 실패: 흉내"}]  # 채점만 실패한 문항: 답변은 남아 있다
    s = evaluate.summarize(results, items)
    check(s["n"] == 3 and s["n_errors"] == 1 and s["errors"] == ["W03"], f"오류 문항 수 {s['n_errors']}, 목록 {s['errors']}")
    check(abs(s["tool_accuracy"] - 0.5) < 1e-9 and abs(s["answer_accuracy"] - 0.5) < 1e-9,
          f"기존 지표는 오류를 뺀 분모 그대로 {s['tool_accuracy']:.3f}/{s['answer_accuracy']:.3f}")
    check(abs(s["tool_accuracy_all"] - 1 / 3) < 1e-9 and abs(s["answer_accuracy_all"] - 1 / 3) < 1e-9,
          f"전체 기준은 오류를 0점으로 센다 {s['tool_accuracy_all']:.3f}/{s['answer_accuracy_all']:.3f}")
    out = tmp / "rep"
    out.mkdir()
    evaluate.write_report(out, "t", s, results, "")
    text = (out / "report.md").read_text(encoding="utf-8")
    check("| 오류 문항 | 1 |" in text and "전체 기준" in text and "평균한 값: work (1칸)" in text
          and "regrade로 다시 채점" in text,
          "보고서에 오류 수·전체 기준 지표·macro F1에 쓴 칸·재채점 안내가 나옴")
    s2 = evaluate.summarize([rec("W01", gold_category=None, pred_category="work")], items)
    check(s2["classification"]["macro_f1"] == 0.0, "정답 칸이 있는 문항이 하나도 없어도 요약이 만들어짐")


def test_history_and_save_run(tmp: Path):
    hist = tmp / "history.csv"
    old_cols = ["일시", "label", "메모", "문항", "분류 정확도", "macro F1", "도구 호출", "답변", "재작성", "차단"]
    old_row = ["2026-11-03 16:17", "01_baseline", "기준", "50", "0.938", "0.942", "0.680", "0.480", "6", "4"]
    with hist.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(old_cols)
        w.writerow(old_row)
    before = hist.read_bytes()
    row = {c: "x" for c in evaluate.HISTORY_COLUMNS}
    evaluate.append_history(hist, row)
    raw = hist.read_bytes()
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    check(raw.startswith(b"\xef\xbb\xbf") and rows[0][:10] == old_cols and rows[0][10:] == evaluate.HISTORY_COLUMNS[10:]
          and len(rows[0]) == len(rows[2]) == 13, f"머리글 끝에만 새 열을 덧붙이고 새 행과 열 수가 같음 {len(rows[0])}/{len(rows[2])}")
    body_old = before.split(b"\r\n", 1)[1]
    check(rows[1] == old_row and body_old in raw, "기존 행은 그대로")
    evaluate.append_history(hist, row)
    rows = list(csv.reader(hist.read_bytes().decode("utf-8-sig").splitlines()))
    check(len(rows) == 4 and rows[0] == rows[0][:13] and len(rows[3]) == 13, "머리글이 이미 갖춰졌으면 다시 고치지 않고 행만 추가")
    fresh = tmp / "fresh.csv"
    evaluate.append_history(fresh, row)
    check(list(csv.reader(fresh.read_text(encoding="utf-8-sig").splitlines()))[0] == evaluate.HISTORY_COLUMNS, "새 파일은 전체 머리글로 시작")

    runs = tmp / "runs_save"
    its = [item("W01"), item("W02")]
    results = [rec("W01"), {**rec("W02"), "error": "채점 실패: 흉내"}]
    with patched(evaluate, RUNS=runs), captured() as out:
        evaluate.save_run("lab", "메모", results, its, 1.0)
    lines = read_jsonl(runs / "lab" / "results.jsonl")
    summ = json.loads((runs / "lab" / "summary.json").read_text(encoding="utf-8"))
    h = list(csv.reader((runs / "history.csv").read_text(encoding="utf-8-sig").splitlines()))
    check(len(lines) == 2 and lines[1]["answer"] == "답" and summ["n_errors"] == 1 and len(h) == 2 and len(h[0]) == len(h[1])
          and "오류 1건" in out(), "save_run: results.jsonl 한 번 기록, 오류 수가 요약·history·출력에 나옴")


def test_print_compare(tmp: Path):
    runs = tmp / "runs_cmp"
    old = {"n": 50, "errors": ["W01"], "classification": {"accuracy": 0.9, "macro_f1": 0.9},
           "tool_accuracy": 0.5, "answer_accuracy": 0.4, "verify_blocked": 1}  # 오류 수·전체 기준을 안 적던 옛 요약
    new = {**old, "n_errors": 2, "errors": ["W01", "W02"], "tool_accuracy_all": 0.4, "answer_accuracy_all": 0.3}
    for name, s in (("a", old), ("b", new)):
        (runs / name).mkdir(parents=True)
        (runs / name / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    with patched(evaluate, RUNS=runs), captured() as out:
        evaluate.print_compare(["a", "b"])
    text = out()
    check("오류 평균 1.5 (범위 1.0~2.0)" in text and "답변(전체) 평균" in text, f"compare가 오류 수를 보여 줌 (옛 요약도 읽음)")


def test_regrade(tmp: Path):
    runs = tmp / "runs_re"
    gold = [item("W01", question="새 질문", category="pay", secondary="work", expected_tools=["lookup_pay", "lookup_work"]),
            item("W03", question="질문"), item("W04", question="질문"), item("W05", question="질문"),
            item("W06", question="질문")]
    gpath = tmp / "gold.json"
    gpath.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    write_jsonl(runs / "src" / "results.jsonl", [
        rec("W01", question="옛 질문"),                       # 정답지가 바뀜(칸·질문)
        rec("W02"),                                          # 지금 정답지에 없음
        rec("W03", answer="BOOM"),                           # 채점이 예외로 실패
        {"id": "W04", "error": "실행 오류 흉내"},               # 에이전트 실행 자체가 실패
        {**rec("W05"), "error": "채점 실패: 전에 났던 오류"},    # 전에 채점만 실패했던 기록
        rec("W06"),
    ])
    args = Namespace(src="src", label="new", strip="", note="")
    with patched(evaluate, RUNS=runs, GOLDENSET=gpath, judge=graded_ok), captured() as out:
        evaluate.cmd_regrade(args)
    text = out()
    res = {r["id"]: r for r in read_jsonl(runs / "new_src" / "results.jsonl")}
    check("W02" not in res and len(res) == 5 and "W02" in text, f"정답지에 없는 문항은 건너뛰고 알림 {sorted(res)}")
    check(res["W01"]["gold_category"] == "pay" and res["W01"]["gold_secondary"] == "work"
          and res["W01"]["expected_tools"] == ["lookup_pay", "lookup_work"] and res["W01"]["tool_score"] == 0,
          "정답 칸·보조 칸·기대 도구가 새 정답지 값으로 갱신됨")
    check("W01" in text and "질문이 정답지와 달라진" in text, "질문이 달라진 문항은 경고 출력")
    check("error" in res["W03"] and res["W03"]["answer"] == "BOOM" and "sk-proj" not in res["W03"]["error"]
          and res["W03"]["tools_called"] == ["lookup_work"], "채점 실패는 그 문항만 error로 남기고 답변 등은 보존")
    check(res["W04"] == {"id": "W04", "error": "실행 오류 흉내"}, "실행 실패 기록은 그대로")
    check("error" not in res["W05"] and res["W05"]["score"] == 1, "전에 채점만 실패했던 기록은 다시 채점돼 오류가 풀림")
    check(res["W06"]["score"] == 1, "나머지 문항은 계속 채점됨")
    summ = json.loads((runs / "new_src" / "summary.json").read_text(encoding="utf-8"))
    check(summ["n"] == 5 and summ["n_errors"] == 2, f"요약: 문항 {summ['n']}, 오류 {summ['n_errors']}")


def test_judge_check(tmp: Path):
    runs = tmp / "runs_jc"
    gpath = tmp / "gold_jc.json"
    gpath.write_text(json.dumps([item("W01"), item("W02")], ensure_ascii=False), encoding="utf-8")
    jpath = tmp / "judge_val.json"
    cases = [{"id": "W01", "answer": "정상", "kind": "good", "expect": 1, "mutation": ""},
             {"id": "W02", "answer": "BOOM", "kind": "good", "expect": 1, "mutation": ""},
             {"id": "W99", "answer": "없는 문항", "kind": "good", "expect": 1, "mutation": ""}]
    jpath.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    with patched(evaluate, RUNS=runs, GOLDENSET=gpath, JUDGE_SET=jpath, judge=graded_ok), captured() as out:
        evaluate.cmd_judge_check(Namespace())
    md = (runs / "judge_check.md").read_text(encoding="utf-8")
    rows = read_jsonl(runs / "judge_check.jsonl")
    check(len(rows) == 3 and sum("error" in r for r in rows) == 2 and "채점 실패(예외): 2건" in md
          and "sk-proj" not in md and "채점 실패 2건" in out(),
          "문항 하나가 실패해도 계속하고 보고서·출력에 실패 수를 적음")
    check("전체 일치: 1/3" in md, "실패한 답안은 판정 일치로 세지 않음")


def test_context():
    import context
    want = {"work": 52, "pay": 46, "welfare": 16, "contract": 47, "harassment": 21}
    got = {c: len(context.lookup(c)) for c in want}
    check(got == want, f"칸별 조각 수 그대로 {got}")
    bad = [ch.id for c in want for ch in context.lookup(c)
           if any(l.startswith(("# ", "## ")) for l in ch.text.splitlines()[1:])]
    check(not bad, f"조각 본문에 장 제목('# '·'## ' 줄)이 안 섞임 {sorted(set(bad))}")
    t = context.company_chunks("docs/company/취업규칙.md")[5].text
    check("제2장" not in t, "취업규칙 제5조 끝에 다음 장 제목이 붙지 않음")


def test_build_goldenset(tmp: Path):
    import build_goldenset as bg
    allowed = bg.load_map()
    base = json.loads((ROOT / "data/goldenset.json").read_text(encoding="utf-8"))[0]
    ok_item = {**base, "must_include": [*base["must_include"], "연차는 15일"], "forbidden_numbers": ["5일"]}
    bad_item = {**base, "must_include": [*base["must_include"], "결혼 휴가는 6일"], "forbidden_numbers": ["6일"]}
    check(bg.check([ok_item], allowed, "goldenset") == [], "금지 '5일'은 정답 문장의 '15일'에 안 걸림(채점과 같은 경계)")
    errs = bg.check([bad_item], allowed, "goldenset")
    check(len(errs) == 1 and "forbidden '6일'" in errs[0], f"정답 문장에 섞인 금지 숫자는 잡음 {errs}")

    def run_main(part_item, name):
        root = tmp / f"bg_{name}"
        (root / "data/_parts").mkdir(parents=True)
        shutil.copytree(ROOT / "docs", root / "docs")
        shutil.copy(ROOT / "data/category_map.json", root / "data/category_map.json")
        (root / "data/examples.json").write_text("[]", encoding="utf-8")
        (root / "data/_parts/part_1.json").write_text(json.dumps([part_item], ensure_ascii=False), encoding="utf-8")
        gp = root / "data/goldenset.json"
        gp.write_text("기존 정답지", encoding="utf-8")
        old_argv = sys.argv
        sys.argv = ["build_goldenset.py"]
        try:
            with patched(bg, ROOT=root), captured():
                code = bg.main()
        finally:
            sys.argv = old_argv
        return code, gp.read_text(encoding="utf-8")

    code, saved = run_main(bad_item, "bad")
    check(code == 1 and saved == "기존 정답지", "검사 위반이 있으면 goldenset.json을 덮어쓰지 않음")
    code, saved = run_main(ok_item, "ok")
    check(code == 0 and saved != "기존 정답지" and json.loads(saved)[0]["id"] == base["id"], "위반이 없으면 저장함")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    test_forbidden_numbers()
    test_classification()
    test_mask_secrets()
    test_judge_merge()
    test_evaluate_item()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_summarize_and_report(tmp)
        test_history_and_save_run(tmp)
        test_print_compare(tmp)
        test_regrade(tmp)
        test_judge_check(tmp)
        test_build_goldenset(tmp)
    test_context()

    if "--judge" in sys.argv:
        ex = next(e for e in json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8")) if e["id"] == "E04")
        good = "네, 알바분도 추석 선물을 받아요. 주 15시간 이상 근무하시면 자사 상품권 7만 원 상당, 15시간 미만이면 3만 원 상당이에요."
        g = grade_answer(ex, good)
        print(f"\n채점 모델 연결: 점수 {g['score']}, 빠짐 {g['missing']}, 위반 {g['violated']}, 금지 숫자 {g['forbidden_hits']}")
        for f in g["judge"]["must_include"]:
            print("  -", "포함" if f["ok"] else "빠짐", f["item"], "|", f["quote"])

    print(f"\n코드 시험 {'모두 통과' if not fails else f'{fails}건 틀림'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
