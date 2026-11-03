"""저장된 답변의 검증 결과를 현재 검증기로 다시 확인한다. 모델 호출 없음."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent
import context


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # 실수로 모델 경로를 사용하면 요청 전에 중단한다.
    def blocked(*args, **kwargs):
        raise RuntimeError("오프라인 검사에서는 모델을 호출할 수 없습니다")
    agent.client = blocked
    seen, failures = set(), []
    for path in sorted((ROOT / "runs").glob("*/results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            log = r.get("verify_log") or []
            if r.get("route") != "answer" or not log or not log[-1].get("ok"):
                continue
            key = (r["question"], r["answer"])
            if key in seen:
                continue
            seen.add(key)
            cats = [c for c in (r.get("pred_category"), r.get("pred_secondary")) if c in context.TOOL_OF]
            chunks = context.assemble(cats)
            check = agent.check_grounding(r["answer"], r.get("citations") or [],
                                          [c.id for c in chunks], context.render(chunks), r["question"])
            if not check["ok"]:
                failures.append((path.parent.name, r, check))
    test = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tests" / "test_verify.py")],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    lines = ["# 저장 답변 검증 회귀 검사", "",
             "모델 호출 없이, 옛 검증기를 통과한 답변을 질문·답변 쌍으로 중복 제거했다.",
             "각 기록의 주 칸·보조 칸으로 현재 근거를 재조립했다.", "",
             f"- 검사한 고유 답변: {len(seen)}건", f"- 새로 막힌 답변: {len(failures)}건", "",
             "| 실행 | 문항 | 문제 |", "|---|---|---|"]
    for run, r, check in failures:
        problem = " / ".join(check["problems"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {run} | {r['id']} | {problem} |")
    lines += ["", "## 새로 막힌 답변 원문", ""]
    for run, r, check in failures:
        lines += [f"### {run} / {r['id']}", "", f"질문: {r['question']}", "",
                  r["answer"], "", f"문제: {' / '.join(check['problems'])}", ""]
    lines += ["## 검증기 시험", "", f"종료 코드: {test.returncode}", "", "```text",
              test.stdout.strip(), test.stderr.strip(), "```", ""]
    out = ROOT / "runs" / "reverify_saved.md"
    if out.exists():
        raise FileExistsError("기존 회귀 결과를 덮어쓸 수 없습니다")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"고유 답변 {len(seen)}건, 새로 막힘 {len(failures)}건. 시험 종료 코드 {test.returncode}")
    for run, r, check in failures:
        print(run, r["id"], check["problems"])
    return test.returncode


if __name__ == "__main__":
    sys.exit(main())
