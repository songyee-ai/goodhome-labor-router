"""실행 묶음별로 실패 원인 건수를 센다(회차 평균). API를 부르지 않는다.

사용: .venv\\Scripts\\python.exe scripts/cause_counts.py 03_gold_v2,03_gold_v2_r2,03_gold_v2_r3 05_strict_secondary_r1,05_strict_secondary_r2,05_strict_secondary_r3
  인자 하나 = 한 실험(쉼표로 회차 나열). 여러 실험을 공백으로 나란히 주면 표 한 장으로 비교한다.
"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
COLS = [
    ("extra_lookup", "필요 없는 보조 칸"),
    ("extra_handoff", "필요 없는 넘김"),
    ("lack_lookup", "빠진 보조 칸"),
    ("miscls", "주 칸 오분류"),
    ("missing", "사실 빠짐"),
    ("violated", "금지 위반"),
    ("forbidden", "금지 숫자"),
    ("blocked", "검증 차단"),
    ("error", "실행 오류"),
]


def count(label: str) -> dict:
    c = dict.fromkeys(k for k, _ in COLS)
    c = {k: 0 for k in c}
    for line in (RUNS / label / "results.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if "error" in r:
            c["error"] += 1
            continue
        extra = set(r["tools_called"]) - set(r["expected_tools"])
        lack = set(r["expected_tools"]) - set(r["tools_called"])
        same_cls = r["gold_category"] == r["pred_category"]
        c["extra_lookup"] += any(t.startswith("lookup") for t in extra)
        c["extra_handoff"] += "handoff" in extra
        c["lack_lookup"] += any(t.startswith("lookup") for t in lack) and same_cls
        c["miscls"] += r["gold_category"] is not None and not same_cls
        c["missing"] += bool(r["missing"])
        c["violated"] += bool(r["violated"])
        c["forbidden"] += bool(r["forbidden_hits"])
        c["blocked"] += bool(r["verify_log"]) and not r["verify_log"][-1]["ok"]
    return c


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("| 실험 | 회차 | " + " | ".join(n for _, n in COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for arg in sys.argv[1:]:
        labels = arg.split(",")
        rows = [count(lb) for lb in labels]
        avg = {k: sum(r[k] for r in rows) / len(rows) for k, _ in COLS}
        name = labels[0].rsplit("_r", 1)[0] if labels[0].rsplit("_r", 1)[-1].isdigit() else labels[0]
        print(f"| {name} | {len(rows)} | " + " | ".join(f"{avg[k]:.1f}" for k, _ in COLS) + " |")


if __name__ == "__main__":
    main()
