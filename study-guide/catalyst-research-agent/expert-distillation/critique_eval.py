r"""Evaluate critique quality on planted-flaw and published-critique test sets.

    # A. Published critiques -> issue lists -> test items
    python critique_eval.py extract-issues --critiques critiques/critiques.jsonl --out critiques/with_issues.jsonl \
        --model qwen3:14b                      # then open the file, check issues, set "checked": true
    python critique_eval.py from-published --critiques critiques/with_issues.jsonl --out published_items.jsonl

    # B. Planted-flaw items come from planted_flaws.py make

    # C. Generate critiques with any OpenAI-compatible model, with and without the rubric
    python critique_eval.py generate --items planted.jsonl --out out_plain.jsonl --model qwen3:14b
    python critique_eval.py generate --items planted.jsonl --out out_rubric.jsonl --model qwen3:14b \
        --rubric-bank rubric_bank_v1.jsonl -k 8

    # D. Score (regex first pass always; LLM judge if --judge-model is given)
    python critique_eval.py score --items planted.jsonl --outputs out_rubric.jsonl --works corpus/works.jsonl \
        --judge-model qwen3:32b --report report_rubric.json

    python critique_eval.py selftest

Default endpoint is a local Ollama server (http://localhost:11434/v1). Published papers may go
to an API model; your own unpublished text should only go to a model you're allowed to use.

Metrics:
    planted_recall        share of planted flaws the critique identifies (judge; regex as fallback)
    false_alarm_rate      on clean controls: share of same-context flaws the critique wrongly claims
    published_recall      share of issues from a published expert critique that the critique raises
    citation_valid_rate   cited DOIs that exist in your corpus (hallucination check)
Judge verdicts count only if the judge quotes evidence that actually appears in the critique.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Any, Callable

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, even under python -I

from planted_flaws import FLAW_BY_ID, FLAWS, detect
from rubric_tools import parse_json_list, quote_in_source, read_jsonl, select, write_jsonl

Chat = Callable[[list[dict[str, str]]], str]
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\)\}>,;\"']+", re.I)

CRITIQUE_PROMPT = """You are reviewing a research plan in zeolite science as a rigorous, constructive expert reviewer.

{rubric_block}Research plan:
\"\"\"
{text}
\"\"\"

Write a critique with these sections:
Strengths: 2-3 bullets.
Weaknesses: numbered, most important first. Each must name the specific problem and what would fix it.{rubric_hint}
Questions: 2-4 questions you would ask the authors.
Cite literature only as doi:... and only if you are sure the DOI exists."""

RUBRIC_BLOCK = "Review criteria (from published reviews in the field):\n{items}\n\n"

ISSUE_JUDGE_PROMPT = """You check whether a critique raises specific issues.

Critique:
\"\"\"
{critique}
\"\"\"

Issues to look for:
{issues}

For each issue, decide whether the critique identifies essentially the same problem (paraphrase is
fine; a vague generic remark is not). Return a JSON list only:
[{{"index": 1, "found": true, "evidence": "exact short quote from the critique"}}, ...]
Use "evidence": "" when found is false."""

EXTRACT_ISSUES_PROMPT = """Below is the abstract (or text) of a published comment criticizing another paper.
List the distinct scientific issues the commenting authors raise about the original paper.

Rules: one issue per item, specific and self-contained; include a verbatim quote (at most 40 words)
from the text supporting it. Return a JSON list only: [{{"issue": ..., "quote": ...}}]

Original paper: "{target_title}"
Comment: "{critique_title}"
Text:
\"\"\"
{text}
\"\"\"
"""


def make_chat(base_url: str, model: str, api_key: str, temperature: float = 0.2) -> Chat:
    def chat(messages: list[dict[str, str]]) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature}
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    return chat


# --------------------------------------------------------------------------------------
# published critiques -> items
# --------------------------------------------------------------------------------------

def extract_issues(rows: list[dict[str, Any]], chat: Chat) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        text = row.get("critique_abstract") or ""
        if len(text) < 100 or row.get("kind") != "comment":
            out.append(row)
            continue
        raw = chat([{"role": "user", "content": EXTRACT_ISSUES_PROMPT.format(
            target_title=row.get("target_title"), critique_title=row.get("critique_title"), text=text)}])
        issues = [i for i in parse_json_list(raw)
                  if str(i.get("issue", "")).strip() and quote_in_source(str(i.get("quote", "")), text)]
        out.append({**row, "issues": issues, "checked": False})
    return out


def items_from_published(rows: list[dict[str, Any]], require_checked: bool) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        if row.get("kind") != "comment" or not row.get("issues") or (require_checked and not row.get("checked")):
            continue
        if len(row.get("target_abstract") or "") < 200:
            continue
        items.append({"id": f"pub-{row['critique_id']}", "kind": "published", "source_doi": row.get("target_doi"),
                      "critique_doi": row.get("critique_doi"),
                      "text": f"Title: {row['target_title']}\n\nAbstract: {row['target_abstract']}",
                      "issues": [i["issue"] for i in row["issues"]]})
    return items


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------

def build_prompt(item: dict[str, Any], bank: list[dict[str, Any]] | None, k: int) -> str:
    if bank:
        rubric = select(bank, item["text"], k)
        lines = "\n".join(f"- [{r['dimension']}] {r['title']}: {r['description']}" for r in rubric)
        return CRITIQUE_PROMPT.format(rubric_block=RUBRIC_BLOCK.format(items=lines), text=item["text"],
                                      rubric_hint=" Where relevant, name the criterion it violates.")
    return CRITIQUE_PROMPT.format(rubric_block="", text=item["text"], rubric_hint="")


def generate(items: list[dict[str, Any]], chat: Chat, bank: list[dict[str, Any]] | None, k: int,
             out: Path, samples: int) -> None:
    done = {(r["id"], r["sample"]) for r in read_jsonl(out)} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for item in items:
            prompt = build_prompt(item, bank, k)
            for s in range(samples):
                if (item["id"], s) in done:
                    continue
                critique = chat([{"role": "user", "content": prompt}])
                f.write(json.dumps({"id": item["id"], "sample": s, "critique": critique,
                                    "with_rubric": bool(bank)}, ensure_ascii=False) + "\n")
                f.flush()


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------

def judge_issues(critique: str, issues: list[str], chat: Chat) -> list[bool]:
    listing = "\n".join(f"{i}. {text}" for i, text in enumerate(issues, 1))
    raw = chat([{"role": "user", "content": ISSUE_JUDGE_PROMPT.format(critique=critique, issues=listing)}])
    found = [False] * len(issues)
    for v in parse_json_list(raw):
        try:
            idx = int(v.get("index")) - 1
        except (TypeError, ValueError):
            continue
        evidence = str(v.get("evidence", ""))
        # judge must point at real text in the critique; short quotes allowed via direct substring check
        grounded = bool(evidence) and (evidence.lower() in critique.lower() or quote_in_source(evidence, critique))
        if 0 <= idx < len(issues) and v.get("found") is True and grounded:
            found[idx] = True
    return found


def citation_check(critique: str, corpus_dois: set[str] | None) -> dict[str, Any]:
    dois = sorted({d.rstrip(".").lower() for d in DOI_RE.findall(critique)})
    if corpus_dois is None:
        return {"cited": len(dois), "invalid": None}
    invalid = [d for d in dois if d not in corpus_dois]
    return {"cited": len(dois), "invalid": len(invalid), "invalid_dois": invalid}


def score_one(item: dict[str, Any], critique: str, chat: Chat | None, corpus_dois: set[str] | None
              ) -> dict[str, Any]:
    res: dict[str, Any] = {"id": item["id"], "kind": item["kind"], **citation_check(critique, corpus_dois)}
    if item["kind"] == "planted" and item["planted"]:
        flaws = [FLAW_BY_ID[p["flaw_id"]] for p in item["planted"]]
        res["planted_regex"] = [detect(f, critique) for f in flaws]
        if chat:
            res["planted_judge"] = judge_issues(critique, [f.description for f in flaws], chat)
    elif item["kind"] == "planted":  # clean control: does it claim flaws that aren't there?
        candidates = [f for f in FLAWS if item["context"] in f.contexts]
        res["false_alarm_regex"] = sum(detect(f, critique) for f in candidates) / len(candidates)
        if chat:
            res["false_alarm_judge"] = sum(judge_issues(critique, [f.description for f in candidates], chat)) / len(candidates)
    elif item["kind"] == "published" and chat:
        res["published_judge"] = judge_issues(critique, item["issues"], chat)
    return res


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        flat = [x for s in scores for x in s.get(key, [])]
        return round(sum(flat) / len(flat), 3) if flat else None

    def mean(key: str) -> float | None:
        vals = [s[key] for s in scores if key in s]
        return round(statistics.mean(vals), 3) if vals else None

    cited = sum(s["cited"] for s in scores)
    invalid = [s["invalid"] for s in scores if s.get("invalid") is not None]
    return {"n_critiques": len(scores),
            "planted_recall_judge": rate("planted_judge"), "planted_recall_regex": rate("planted_regex"),
            "false_alarm_judge": mean("false_alarm_judge"), "false_alarm_regex": mean("false_alarm_regex"),
            "published_recall_judge": rate("published_judge"),
            "dois_cited": cited,
            "citation_valid_rate": round(1 - sum(invalid) / cited, 3) if invalid and cited else None}


# --------------------------------------------------------------------------------------

def selftest() -> None:
    from planted_flaws import build_item
    rec = {"title": "MFI nanosheet membranes for xylene separation", "doi": "10.0000/x.1",
           "abstract": "Zeolite membranes can save energy. Thin layers matter. More text here. " * 8}
    item = build_item(rec, [FLAW_BY_ID["mismatched_benchmark"]], "t1")
    clean = build_item(rec, [], "t2")
    good = ("Weaknesses: 1. The plan compares separation factors with literature values measured at different "
            "feed pressure and temperature; comparisons must be at matched conditions (see doi:10.0000/x.1). "
            "2. See doi:10.9999/fake.123.")

    def fake_judge(messages):
        prompt = messages[0]["content"]
        n = len(re.findall(r"^\d+\. ", prompt.split("Issues to look for:")[1], re.M))
        verdicts = [{"index": i + 1, "found": False, "evidence": ""} for i in range(n)]
        if "matched conditions" in prompt and "Benchmarking against literature" in prompt:
            idx = prompt.split("Issues to look for:")[1].split("\n").index(
                next(l for l in prompt.split("Issues to look for:")[1].split("\n") if "Benchmarking" in l))
            verdicts[idx - 1] = {"index": idx, "found": True, "evidence": "comparisons must be at matched conditions"}
        if n > 1:  # hallucinated verdict without real evidence must be ignored
            verdicts[-1] = {"index": n, "found": True, "evidence": "this sentence is not in the critique at all"}
        return json.dumps(verdicts)

    s1 = score_one(item, good, fake_judge, {"10.0000/x.1"})
    assert s1["planted_regex"] == [True] and s1["planted_judge"] == [True], s1
    assert s1["cited"] == 2 and s1["invalid"] == 1, s1
    s2 = score_one(clean, "Looks solid overall.", fake_judge, None)
    assert s2["false_alarm_judge"] == 0.0, s2
    summary = summarize([s1, s2])
    assert summary["planted_recall_judge"] == 1.0 and summary["citation_valid_rate"] == 0.5, summary
    assert "Review criteria" in build_prompt(item, [{"kind": "rubric", "title": "Fair benchmarking",
                                                     "description": "Compare at matched conditions.",
                                                     "dimension": "performance_benchmarking", "importance": "high",
                                                     "sources": [{"doi": "10.0000/x.2"}]}], 4)
    pub = items_from_published([{"critique_id": "W10", "kind": "comment", "target_title": "T", "target_doi": "d",
                                 "critique_doi": "c", "target_abstract": "a" * 250, "checked": True,
                                 "issues": [{"issue": "Thickness not measured", "quote": "q"}]}], True)
    assert pub[0]["issues"] == ["Thickness not measured"]
    print("selftest passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_args(s, prefix=""):
        s.add_argument(f"--{prefix}base-url", default="http://localhost:11434/v1")
        s.add_argument(f"--{prefix}model", required=(prefix == ""))
        s.add_argument(f"--{prefix}api-key", default="local")

    s = sub.add_parser("extract-issues")
    s.add_argument("--critiques", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    llm_args(s)
    s = sub.add_parser("from-published")
    s.add_argument("--critiques", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--include-unchecked", action="store_true")
    s = sub.add_parser("generate")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rubric-bank", type=Path)
    s.add_argument("-k", type=int, default=8)
    s.add_argument("--samples", type=int, default=1)
    s.add_argument("--temperature", type=float, default=0.7)
    llm_args(s)
    s = sub.add_parser("score")
    s.add_argument("--items", type=Path, required=True)
    s.add_argument("--outputs", type=Path, required=True)
    s.add_argument("--works", type=Path, help="corpus works.jsonl for the citation check")
    s.add_argument("--report", type=Path, required=True)
    llm_args(s, "judge-")
    sub.add_parser("selftest")
    args = p.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "extract-issues":
        rows = extract_issues(read_jsonl(args.critiques), make_chat(args.base_url, args.model, args.api_key))
        write_jsonl(args.out, rows)
        print(f"wrote {args.out}; now check each 'issues' list and set \"checked\": true")
    elif args.cmd == "from-published":
        items = items_from_published(read_jsonl(args.critiques), not args.include_unchecked)
        write_jsonl(args.out, items)
        print(f"wrote {len(items)} published-critique items to {args.out}")
    elif args.cmd == "generate":
        bank = read_jsonl(args.rubric_bank) if args.rubric_bank else None
        generate(read_jsonl(args.items), make_chat(args.base_url, args.model, args.api_key, args.temperature),
                 bank, args.k, args.out, args.samples)
    elif args.cmd == "score":
        items = {i["id"]: i for i in read_jsonl(args.items)}
        chat = make_chat(args.judge_base_url, args.judge_model, args.judge_api_key, 0.0) if args.judge_model else None
        corpus = ({(r.get("doi") or "").lower() for r in read_jsonl(args.works) if r.get("doi")}
                  if args.works else None)
        scores = [score_one(items[o["id"]], o["critique"], chat, corpus)
                  for o in read_jsonl(args.outputs) if o["id"] in items]
        report = {"summary": summarize(scores), "per_item": scores}
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
