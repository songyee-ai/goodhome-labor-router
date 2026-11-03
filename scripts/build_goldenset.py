"""data/_parts/*.json을 합쳐 data/goldenset.json을 만들고 규격 위반을 검사한다.

사용: python scripts/build_goldenset.py [--check-only]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import forbidden_hits  # noqa: E402  채점과 같은 경계 규칙으로 금지 숫자를 찾는다

TOOLS = {
    "work": "lookup_work",
    "pay": "lookup_pay",
    "welfare": "lookup_welfare",
    "contract": "lookup_contract",
    "harassment": "lookup_harassment",
}
PREFIX = {"work": "W", "pay": "P", "welfare": "F", "contract": "C",
          "harassment": "H", "out_of_scope": "O", None: "X"}
FIELDS = ["id", "persona", "question", "category", "secondary", "expected_tools", "type",
          "must_include", "must_not_include", "forbidden_numbers", "evidence", "note"]


def load_map():
    m = json.loads((ROOT / "data/category_map.json").read_text(encoding="utf-8"))
    allowed = {}
    for cid, c in m["categories"].items():
        refs = set()
        for src in c["company"]:
            doc = "취업규칙" if "취업규칙" in src["file"] else "복리후생규정"
            for s in src["articles"]:
                for n in (range(s[0], s[1] + 1) if isinstance(s, list) else [s]):
                    refs.add(f"{doc} 제{n}조")
        for p in c["law"]:
            text = (ROOT / p).read_text(encoding="utf-8")
            refs.update(re.findall(r"(?m)^## (.+? 제\d+조(?:의\d+)?)", text))
            # 조문이 아닌 절(예: "2026년 적용 최저임금")도 근거로 인정한다
            refs.update(h for h in re.findall(r"(?m)^## (.+)$", text) if "조(" not in h)
        allowed[cid] = refs
    return allowed


def ref_key(ev):
    # "근로기준법 제60조 제1항" → "근로기준법 제60조", "취업규칙 제28조 ①" → "취업규칙 제28조"
    m = re.match(r"(.+? 제\d+조(?:의\d+)?)", ev.strip())
    return m.group(1) if m else ev.strip()


def check(items, allowed, label):
    errors = []
    ids = Counter(i.get("id") for i in items)
    errors += [f"{k}: id 중복" for k, v in ids.items() if v > 1]
    for it in items:
        iid = it.get("id", "?")
        missing = [f for f in FIELDS if f not in it]
        if missing:
            errors.append(f"{iid}: 필드 없음 {missing}")
            continue
        cat, sec = it["category"], it["secondary"]
        if label == "goldenset" and not iid.startswith(PREFIX.get(cat, "?")):
            errors.append(f"{iid}: id 접두어가 category({cat})와 안 맞음")
        want = set()
        if cat in TOOLS:
            want.add(TOOLS[cat])
        if sec in TOOLS:
            want.add(TOOLS[sec])
        if cat in ("out_of_scope", None) or it["type"] == "partial_handoff":
            want.add("handoff")
        if set(it["expected_tools"]) != want:
            errors.append(f"{iid}: expected_tools {it['expected_tools']} ≠ 규격상 {sorted(want)}")
        scope = allowed.get(cat, set()) | allowed.get(sec, set())
        for ev in it["evidence"]:
            if ref_key(ev) not in scope and not any(ev.startswith(h) for h in scope):
                errors.append(f"{iid}: evidence '{ev}'가 {cat}/{sec} 근거 범위 밖")
        for num in it["forbidden_numbers"]:
            if any(forbidden_hits(fact, [num]) for fact in it["must_include"]):
                errors.append(f"{iid}: forbidden '{num}'이 must_include 안에 들어 있음")
        if not it["must_include"]:
            errors.append(f"{iid}: must_include 비어 있음")
    return errors


def summary(items):
    print("카테고리:", dict(Counter(str(i["category"]) for i in items)))
    print("type:", dict(Counter(i["type"] for i in items)))
    print("persona:", dict(Counter(i["persona"] for i in items)))
    print("보조 칸 있음:", sum(1 for i in items if i["secondary"]))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    allowed = load_map()
    parts = sorted((ROOT / "data/_parts").glob("part_*.json"))
    items = [x for p in parts for x in json.loads(p.read_text(encoding="utf-8"))]
    order = "WPFCHOX"
    items.sort(key=lambda i: (order.index(i["id"][0]), i["id"]))
    examples = json.loads((ROOT / "data/examples.json").read_text(encoding="utf-8"))

    print(f"평가 문항 {len(items)}건 ({', '.join(p.name for p in parts)})")
    summary(items)
    errs = check(items, allowed, "goldenset") + check(examples, allowed, "examples")
    eq = {i["question"] for i in items} & {e["question"] for e in examples}
    errs += [f"예시와 평가가 같은 질문: {q}" for q in eq]
    print(f"\n예시 {len(examples)}건")
    summary(examples)
    print("\n검사 결과:", "통과" if not errs else f"{len(errs)}건")
    for e in errs:
        print(" -", e)
    if "--check-only" in sys.argv:
        pass
    elif errs:  # 위반이 있는 채로 저장하면 평가가 잘못된 정답지로 돌아간다
        print("\n검사 위반이 있어 data/goldenset.json을 저장하지 않았습니다(기존 파일 그대로).")
    else:
        (ROOT / "data/goldenset.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n저장: data/goldenset.json")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
