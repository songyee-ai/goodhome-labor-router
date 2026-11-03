"""근거 문서를 조(條) 단위로 쪼개고, 카테고리별로 필요한 조만 조립한다.

- 사규(docs/company/*.md): "### 제N조(제목)" 단위. 조각 id는 "취업규칙 제28조"처럼 문서 이름 + 조 번호.
- 법령(docs/law/*.md): "## 근로기준법 제60조(제목)" 단위. 조각 id는 제목 줄에서 괄호 앞까지.
  조문이 아닌 절(예: "## 2026년 적용 최저임금")도 그 제목을 id로 한 조각이 된다.
- 어느 카테고리에 어떤 조각을 넣을지는 data/category_map.json이 정한다.
"""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORY_MAP = ROOT / "data" / "category_map.json"

# 조회 도구 이름. 평가셋의 expected_tools와 같은 이름을 쓴다.
TOOL_OF = {
    "work": "lookup_work",
    "pay": "lookup_pay",
    "welfare": "lookup_welfare",
    "contract": "lookup_contract",
    "harassment": "lookup_harassment",
}
HANDOFF_TOOL = "handoff"


@dataclass(frozen=True)
class Chunk:
    id: str      # 예: "취업규칙 제28조", "근로기준법 제60조", "2026년 적용 최저임금"
    title: str   # 제목 줄 전체
    text: str    # 제목 줄을 포함한 원문
    source: str  # 파일 경로(프로젝트 기준)


def _split(path: Path, level: str) -> list[tuple[str, str]]:
    """'### ' 또는 '## ' 제목 줄 기준으로 (제목, 본문) 목록을 만든다.

    본문은 다음 장 제목('# ', '## ' 줄)이 나오면 거기서 끊는다. 안 그러면 '## 제2장 …'이나
    '## 부칙' 같은 장 제목이 바로 앞 조의 본문 끝에 붙는다.
    """
    text = path.read_text(encoding="utf-8")
    parts = re.split(rf"(?m)^{level} ", text)
    out = []
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        body = re.split(r"(?m)^#{1,2} ", body)[0]
        out.append((title.strip(), body.strip()))
    return out


@lru_cache(maxsize=None)
def company_chunks(rel_path: str) -> dict[int, Chunk]:
    """사규 파일의 조 번호 → 조각."""
    path = ROOT / rel_path
    doc = path.stem  # "취업규칙", "복리후생규정"
    chunks = {}
    for title, body in _split(path, "###"):
        m = re.match(r"제(\d+)조", title)
        if m:
            n = int(m.group(1))
            chunks[n] = Chunk(f"{doc} 제{n}조", title, f"[{doc}] {title}\n{body}", rel_path)
    return chunks


@lru_cache(maxsize=None)
def law_chunks(rel_path: str) -> list[Chunk]:
    """법령 근거 파일의 조문(과 절) 조각."""
    path = ROOT / rel_path
    chunks = []
    for title, body in _split(path, "##"):
        cid = title.split("(")[0].strip()
        chunks.append(Chunk(cid, title, f"{title}\n{body}", rel_path))
    return chunks


@lru_cache(maxsize=None)
def category_map() -> dict:
    return json.loads(CATEGORY_MAP.read_text(encoding="utf-8"))["categories"]


def _expand(spec) -> list[int]:
    nums = []
    for s in spec:
        nums.extend(range(s[0], s[1] + 1) if isinstance(s, list) else [s])
    return nums


def lookup(category: str) -> list[Chunk]:
    """조회 도구 하나(lookup_<category>)가 돌려주는 근거 조각 목록."""
    c = category_map()[category]
    chunks = []
    for src in c["company"]:
        table = company_chunks(src["file"])
        chunks.extend(table[n] for n in _expand(src["articles"]) if n in table)
    for p in c["law"]:
        chunks.extend(law_chunks(p))
    return chunks


def assemble(categories: list[str]) -> list[Chunk]:
    """여러 카테고리의 근거를 합친다. 같은 조각은 한 번만 넣는다."""
    seen, out = set(), []
    for cat in categories:
        for ch in lookup(cat):
            if ch.id not in seen:
                seen.add(ch.id)
                out.append(ch)
    return out


def render(chunks: list[Chunk]) -> str:
    """프롬프트에 넣을 근거 문자열. 조각마다 [근거 id]를 앞에 붙인다."""
    return "\n\n".join(f"<근거 id=\"{c.id}\">\n{c.text}\n</근거>" for c in chunks)


def handoff_contact() -> str:
    return category_map()["out_of_scope"]["handoff"]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for cat in TOOL_OF:
        chunks = lookup(cat)
        size = sum(len(c.text) for c in chunks)
        print(f"{cat:11} 조각 {len(chunks):3}개 {size:7,}자  예: {chunks[0].id} … {chunks[-1].id}")
    both = assemble(["work", "welfare"])
    print("work+welfare 합치면", len(both), "개(겹침 제거)")
