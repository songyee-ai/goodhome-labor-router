"""문항 하나를 분류 → 근거 조립 → 답변까지만 돌려 검증 전 초안과 검증 결과를 보여 준다(디버깅용).

사용: .venv\\Scripts\\python.exe scripts/debug_draft.py F03
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import agent  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print("사용: .venv\\Scripts\\python.exe scripts/debug_draft.py <문항 id 또는 질문 문장>\n"
              "  예: scripts/debug_draft.py F03")
        return 1
    gold = {i["id"]: i for i in json.loads((ROOT / "data/goldenset.json").read_text(encoding="utf-8"))}
    q = gold[sys.argv[1]]["question"] if sys.argv[1] in gold else " ".join(sys.argv[1:])
    s = {"question": q}
    for step in (agent.classify, agent.gate, agent.retrieve, agent.answer, agent.verify):
        s.update(step(s))
    print("질문:", q)
    print("분류:", s["category"], s.get("secondary"), s["confidence"])
    print("초안:\n" + s["draft"])
    print("uncovered:", s["uncovered"])
    print("검증:", s["verify"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
