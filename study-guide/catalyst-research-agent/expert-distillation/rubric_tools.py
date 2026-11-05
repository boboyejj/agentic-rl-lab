r"""Turn an expert corpus into grading rubrics, critique questions and expert-annotation sheets.

Standard library only. Works with the outputs of build_expert_corpus.py.

    # 1. Build extraction jobs (one prompt per review/perspective) for a teacher model.
    python rubric_tools.py prompts --docs corpus/opinion_docs.jsonl --out jobs.jsonl \
        [--fulltext-dir corpus/text]          # optional: <sanitized-doi>.txt files you may legally store

    # 2. Run the jobs on any OpenAI-compatible endpoint. Default: a *local* Ollama server,
    #    so nothing leaves your machine. (vLLM: --base-url http://localhost:8000/v1)
    python rubric_tools.py run --jobs jobs.jsonl --out raw.jsonl --model qwen3:14b

    # 3. Validate (quotes must appear in the source), deduplicate, and build the rubric bank.
    python rubric_tools.py merge --jobs jobs.jsonl --raw raw.jsonl --out rubric_bank.jsonl

    # 4. Pick the rubric items relevant to one task, in Tournament-GRPO's judge format.
    python rubric_tools.py select --bank rubric_bank.jsonl --query "Aim 2: OSDA-free MFI nanosheets" -k 8

    # 5. Blinded A/B sheet for you (and optionally a colleague), then judge-vs-human agreement.
    python rubric_tools.py pairs --a sysA.jsonl --b sysB.jsonl --out pairs.csv
    python rubric_tools.py agreement --sheet pairs_filled.csv --key pairs.key.json --judge judge.jsonl

    python rubric_tools.py selftest

Every rubric item keeps its source DOI and a verbatim quote. Items whose quote cannot be found
in the source text are dropped, so the bank contains only criteria grounded in published work.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

# A fixed vocabulary keeps rubric items comparable across experts. Edit for your field.
DIMENSIONS = {
    "characterization_evidence": "Is each structural/morphological claim backed by the right measurements "
                                 "(XRD, TEM/SEM, AFM, N2/Ar physisorption, NMR, ICP...)?",
    "synthesis_reproducibility": "Are gel composition, reagents, aging, temperature/time, seeds and yields "
                                 "reported well enough to repeat the synthesis?",
    "mechanism_claims": "Are proposed formation/crystallization mechanisms supported, with alternatives "
                        "considered?",
    "structure_property_link": "Is the link between structure (thickness, defects, Si/Al, orientation) and "
                               "performance demonstrated rather than assumed?",
    "performance_benchmarking": "Is performance compared fairly with the state of the art under comparable "
                                "conditions (separation factor, permeance, stability, conversion/selectivity)?",
    "scalability_cost": "Are scale-up, cost, OSDA use, reproducibility across batches, and long-term "
                        "stability addressed?",
    "novelty_positioning": "Is the contribution clearly positioned against prior work?",
    "safety_sustainability": "Are hazards, waste (e.g. OSDA, HF) and sustainability considered?",
}
IMPORTANCE = {"high", "medium", "low"}

RUBRIC_PROMPT = """You extract evaluation criteria from a published {role} by {experts}.

Goal: criteria an expert reviewer in this field would apply when judging a paper or research
proposal on {themes}. Use ONLY what the text below states or directly implies.

Rules:
- Every item must include a verbatim quote (at most 40 words) copied exactly from the text.
- Phrase the description as a check a reviewer can apply ("Does the work ...?").
- dimension must be one of: {dimensions}.
- importance: "high" if the authors call it critical/essential/a key challenge; "medium" if
  recommended; "low" otherwise.
- Do not invent opinions. If the text supports no criteria, return [].
- Return a JSON list only: [{{"title": ..., "description": ..., "dimension": ..., "importance": ...,
  "quote": ..., "location": "abstract|introduction|section name"}}]

Source: doi:{doi} ({year}) "{title}"
Text ({text_kind}):
\"\"\"
{text}
\"\"\"
"""

QUESTIONS_PROMPT = """From the published {role} below by {experts}, list the questions an expert
reviewer would ask about a NEW paper or proposal on {themes}, based on the problems, pitfalls and
open challenges this text identifies.

Rules:
- Each question must be supported by a verbatim quote (at most 40 words) from the text.
- Questions must be specific and answerable ("Is nanosheet thickness measured by AFM or TEM
  rather than inferred from XRD peak broadening?"), not generic ("Is the work novel?").
- dimension must be one of: {dimensions}.
- Return a JSON list only: [{{"question": ..., "dimension": ..., "quote": ..., "location": ...}}]

Source: doi:{doi} ({year}) "{title}"
Text ({text_kind}):
\"\"\"
{text}
\"\"\"
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_doi(doi: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", doi)


def normalize(text: str) -> str:
    text = text.lower().replace("‐", "-").replace("‑", "-").replace("–", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%./-]", " ", text)).strip()


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if len(t) > 2]


# --------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------

def source_text(doc: dict[str, Any], fulltext_dir: Path | None, max_chars: int) -> tuple[str, str]:
    if fulltext_dir and doc.get("doi"):
        path = fulltext_dir / f"{sanitize_doi(doc['doi'])}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars], "full text (truncated)"
    return doc.get("abstract") or "", "abstract only"


def build_jobs(docs: list[dict[str, Any]], fulltext_dir: Path | None, themes: str,
               max_chars: int, kinds: list[str]) -> list[dict[str, Any]]:
    jobs = []
    for doc in docs:
        text, text_kind = source_text(doc, fulltext_dir, max_chars)
        if len(text) < 200:
            continue  # nothing to extract from
        fields = dict(role="review/perspective" if doc.get("role") == "opinion" else doc.get("role", "work"),
                      experts=", ".join(doc.get("experts", [])) or "the authors", themes=themes,
                      dimensions=", ".join(DIMENSIONS), doi=doc.get("doi") or doc.get("openalex_id"),
                      year=doc.get("year"), title=doc.get("title"), text_kind=text_kind, text=text)
        for kind in kinds:
            template = RUBRIC_PROMPT if kind == "rubric" else QUESTIONS_PROMPT
            jobs.append({"job_id": f"{kind}:{fields['doi']}", "kind": kind, "doi": fields["doi"],
                         "experts": doc.get("experts", []), "source_text": text,
                         "messages": [{"role": "user", "content": template.format(**fields)}]})
    return jobs


# --------------------------------------------------------------------------------------
# run (OpenAI-compatible chat endpoint; local by default)
# --------------------------------------------------------------------------------------

def run_jobs(jobs: list[dict[str, Any]], base_url: str, model: str, api_key: str, out: Path) -> None:
    done = {r["job_id"] for r in read_jsonl(out)} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for job in jobs:
            if job["job_id"] in done:
                continue
            body = {"model": model, "messages": job["messages"], "temperature": 0.2}
            req = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                content = json.loads(resp.read())["choices"][0]["message"]["content"]
            f.write(json.dumps({"job_id": job["job_id"], "model": model, "response": content}) + "\n")
            f.flush()
            print(f"[ok] {job['job_id']}")


# --------------------------------------------------------------------------------------
# merge: validate + dedupe
# --------------------------------------------------------------------------------------

def parse_json_list(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def quote_in_source(quote: str, source: str) -> bool:
    q, s = normalize(quote), normalize(source)
    if len(q) < 15:
        return False
    if q in s:
        return True
    # tolerate small OCR/whitespace differences: best fuzzy window match
    matcher = difflib.SequenceMatcher(None, s, q, autojunk=False)
    block = matcher.find_longest_match(0, len(s), 0, len(q))
    return block.size >= 0.8 * len(q)


def validate_item(item: dict[str, Any], kind: str, source: str) -> str | None:
    """Return a rejection reason, or None if the item is valid."""
    text_field = "description" if kind == "rubric" else "question"
    required = ["title", "description", "dimension", "importance", "quote"] if kind == "rubric" \
        else ["question", "dimension", "quote"]
    for key in required:
        if not str(item.get(key, "")).strip():
            return f"missing {key}"
    if item["dimension"] not in DIMENSIONS:
        return f"bad dimension {item['dimension']!r}"
    if kind == "rubric" and str(item["importance"]).lower() not in IMPORTANCE:
        return f"bad importance {item['importance']!r}"
    if len(str(item["quote"]).split()) > 60:
        return "quote too long"
    if not quote_in_source(str(item["quote"]), source):
        return "quote not found in source"
    if len(str(item[text_field]).split()) < 5:
        return "too vague"
    return None


def dedupe(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        text = normalize(item.get("description") or item.get("question") or "")
        for other in kept:
            other_text = normalize(other.get("description") or other.get("question") or "")
            if other["kind"] == item["kind"] and \
                    difflib.SequenceMatcher(None, text, other_text).ratio() >= threshold:
                other["sources"].extend(item["sources"])
                other["support"] = len({s["doi"] for s in other["sources"]})
                break
        else:
            kept.append(item)
    return kept


def merge(jobs: list[dict[str, Any]], raw: list[dict[str, Any]], threshold: float
          ) -> tuple[list[dict[str, Any]], Counter]:
    by_id = {j["job_id"]: j for j in jobs}
    reasons: Counter = Counter()
    items = []
    for row in raw:
        job = by_id.get(row["job_id"])
        if not job:
            reasons["unknown job"] += 1
            continue
        for item in parse_json_list(row["response"]):
            reason = validate_item(item, job["kind"], job["source_text"])
            if reason:
                reasons[reason] += 1
                continue
            reasons["accepted"] += 1
            entry = {"kind": job["kind"], "dimension": item["dimension"],
                     "sources": [{"doi": job["doi"], "experts": job["experts"], "quote": item["quote"],
                                  "location": item.get("location", "")}], "support": 1,
                     "extracted_by": row.get("model"), "expert_reviewed": False}
            if job["kind"] == "rubric":
                entry.update(title=item["title"].strip(), description=item["description"].strip(),
                             importance=str(item["importance"]).lower())
            else:
                entry.update(question=item["question"].strip())
            items.append(entry)
    bank = dedupe(items, threshold)
    for i, item in enumerate(bank):
        item["id"] = f"{item['kind'][:1]}{i:04d}"
    return bank, reasons


# --------------------------------------------------------------------------------------
# select: per-task rubric in Tournament-GRPO's format
# --------------------------------------------------------------------------------------

def to_tournament(item: dict[str, Any]) -> dict[str, str]:
    """experiments/tournament-grpo/judge.py:build_judge_context expects these four string keys."""
    cite = "; ".join(sorted({f"doi:{s['doi']}" for s in item["sources"]}))
    return {"title": item["title"], "description": f"{item['description']} (source: {cite})",
            "dimension": item["dimension"], "importance": item["importance"]}


def select(bank: list[dict[str, Any]], query: str, k: int, per_dimension: int = 2) -> list[dict[str, str]]:
    rubrics = [b for b in bank if b["kind"] == "rubric"]
    df = Counter(t for b in rubrics for t in set(tokens(b["title"] + " " + b["description"])))
    q = Counter(tokens(query))
    weight = {"high": 1.5, "medium": 1.0, "low": 0.7}

    def score(b: dict[str, Any]) -> float:
        doc = Counter(tokens(b["title"] + " " + b["description"]))
        overlap = sum(min(q[t], doc[t]) * math.log(1 + len(rubrics) / (1 + df[t])) for t in q)
        return (overlap + 0.1 * math.log1p(b.get("support", 1))) * weight.get(b["importance"], 1.0)

    chosen, per_dim = [], Counter()
    for b in sorted(rubrics, key=score, reverse=True):
        if per_dim[b["dimension"]] < per_dimension:
            chosen.append(b)
            per_dim[b["dimension"]] += 1
        if len(chosen) == k:
            break
    return [to_tournament(b) for b in chosen]


# --------------------------------------------------------------------------------------
# pairs + agreement: efficient expert involvement
# --------------------------------------------------------------------------------------

def make_pairs(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], seed: int
               ) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """Inputs: JSONL with {"id", "query", "answer"} from two systems. Output: blinded sheet + key."""
    rng = random.Random(seed)
    b_by_id = {r["id"]: r for r in b_rows}
    sheet, key = [], {}
    for a in a_rows:
        b = b_by_id.get(a["id"])
        if not b:
            continue
        left_is_a = rng.random() < 0.5
        left, right = (a, b) if left_is_a else (b, a)
        sheet.append({"id": a["id"], "query": a["query"], "response_1": left["answer"],
                      "response_2": right["answer"], "better (1/2/tie)": "",
                      "confidence (1-3)": "", "one-line reason": ""})
        key[a["id"]] = {"1": "A" if left_is_a else "B", "2": "B" if left_is_a else "A"}
    rng.shuffle(sheet)
    return sheet, key


def cohen_kappa(x: list[str], y: list[str]) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[c] * cy[c] for c in set(x) | set(y)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(sheet: list[dict[str, str]], key: dict[str, dict[str, str]],
              judge: dict[str, str]) -> dict[str, float]:
    """judge: {id: "A"|"B"|"tie"}; sheet rows carry the expert's '1'/'2'/'tie'."""
    expert, model = [], []
    for row in sheet:
        choice = row["better (1/2/tie)"].strip().lower()
        if not choice or row["id"] not in judge:
            continue
        expert.append("tie" if choice == "tie" else key[row["id"]][choice])
        model.append(judge[row["id"]])
    n = len(expert)
    return {"n": n, "agreement": sum(a == b for a, b in zip(expert, model)) / n if n else float("nan"),
            "cohen_kappa": cohen_kappa(expert, model),
            "expert_prefers_A": sum(e == "A" for e in expert) / n if n else float("nan")}


# --------------------------------------------------------------------------------------
# selftest (synthetic text, no real papers)
# --------------------------------------------------------------------------------------

def selftest() -> None:
    abstract = ("Two-dimensional zeolites promise short diffusion paths, but reported thickness values are "
                "often inferred from XRD peak broadening alone. We argue that thickness must be confirmed "
                "by AFM or TEM on many sheets, and that membrane separation factors should be compared at "
                "matched feed pressure and temperature. Scale-up of OSDA-intensive routes remains a key "
                "challenge for practical membranes.")
    docs = [{"doi": "10.0000/example.2", "role": "opinion", "experts": ["one"], "year": 2022,
             "title": "Perspective on 2D zeolites", "abstract": abstract}]
    jobs = build_jobs(docs, None, "zeolite nanosheets and membranes", 20000, ["rubric", "questions"])
    assert len(jobs) == 2 and "doi:10.0000/example.2" in jobs[0]["messages"][0]["content"]
    rubric_resp = json.dumps([
        {"title": "Direct thickness measurement", "description": "Does the work confirm nanosheet "
         "thickness by AFM or TEM rather than XRD broadening alone?", "dimension": "characterization_evidence",
         "importance": "high", "quote": "thickness must be confirmed by AFM or TEM on many sheets"},
        {"title": "Same, reworded", "description": "Does the work confirm nanosheet thickness by AFM or "
         "TEM rather than by XRD broadening alone?", "dimension": "characterization_evidence",
         "importance": "high", "quote": "thickness must be confirmed by AFM or TEM"},
        {"title": "Invented", "description": "Does the work use machine learning to design OSDAs?",
         "dimension": "novelty_positioning", "importance": "high", "quote": "ML-designed OSDAs are essential"},
        {"title": "Fair benchmarking", "description": "Are separation factors compared at matched feed "
         "pressure and temperature?", "dimension": "performance_benchmarking", "importance": "medium",
         "quote": "membrane separation factors should be compared at matched feed pressure and temperature"},
    ])
    q_resp = "<think>...</think>" + json.dumps([
        {"question": "Is OSDA consumption per gram of membrane reported and is the route scalable?",
         "dimension": "scalability_cost", "quote": "Scale-up of OSDA-intensive routes remains a key challenge"}])
    raw = [{"job_id": jobs[0]["job_id"], "model": "teacher", "response": rubric_resp},
           {"job_id": jobs[1]["job_id"], "model": "teacher", "response": q_resp}]
    bank, reasons = merge(jobs, raw, threshold=0.85)
    assert reasons["quote not found in source"] == 1, reasons
    assert sum(b["kind"] == "rubric" for b in bank) == 2, bank  # duplicate merged, invented dropped
    picked = select(bank, "How should we characterize nanosheet thickness?", k=1)
    assert picked[0]["title"] == "Direct thickness measurement" and "doi:10.0000" in picked[0]["description"]
    sheet, key = make_pairs([{"id": "q1", "query": "q", "answer": "a"}, {"id": "q2", "query": "q", "answer": "a2"}],
                            [{"id": "q1", "query": "q", "answer": "b"}, {"id": "q2", "query": "q", "answer": "b2"}], 0)
    for row in sheet:
        row["better (1/2/tie)"] = next(k for k, v in key[row["id"]].items() if v == "A")
    stats = agreement(sheet, key, {"q1": "A", "q2": "A"})
    assert stats["agreement"] == 1.0 and stats["n"] == 2, stats
    print("selftest passed:", dict(reasons))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prompts")
    s.add_argument("--docs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--fulltext-dir", type=Path)
    s.add_argument("--themes", default="zeolite synthesis, 2D zeolites/nanosheets and zeolite membranes")
    s.add_argument("--max-chars", type=int, default=24000)
    s.add_argument("--kinds", default="rubric,questions")
    s = sub.add_parser("run")
    s.add_argument("--jobs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--base-url", default="http://localhost:11434/v1")
    s.add_argument("--model", required=True)
    s.add_argument("--api-key", default="local")
    s = sub.add_parser("merge")
    s.add_argument("--jobs", type=Path, required=True)
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--dedupe-threshold", type=float, default=0.85)
    s = sub.add_parser("select")
    s.add_argument("--bank", type=Path, required=True)
    s.add_argument("--query", required=True)
    s.add_argument("-k", type=int, default=8)
    s = sub.add_parser("pairs")
    s.add_argument("--a", type=Path, required=True)
    s.add_argument("--b", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("agreement")
    s.add_argument("--sheet", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--judge", type=Path, required=True, help='JSONL of {"id", "winner": "A"|"B"|"tie"}')
    sub.add_parser("selftest")
    args = p.parse_args()

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "prompts":
        jobs = build_jobs(read_jsonl(args.docs), args.fulltext_dir, args.themes, args.max_chars,
                          args.kinds.split(","))
        write_jsonl(args.out, jobs)
        print(f"wrote {len(jobs)} jobs to {args.out}")
    elif args.cmd == "run":
        run_jobs(read_jsonl(args.jobs), args.base_url, args.model, args.api_key, args.out)
    elif args.cmd == "merge":
        bank, reasons = merge(read_jsonl(args.jobs), read_jsonl(args.raw), args.dedupe_threshold)
        write_jsonl(args.out, bank)
        print(f"wrote {len(bank)} items to {args.out}; validation: {dict(reasons)}")
    elif args.cmd == "select":
        json.dump(select(read_jsonl(args.bank), args.query, args.k), sys.stdout, indent=2, ensure_ascii=False)
        print()
    elif args.cmd == "pairs":
        sheet, key = make_pairs(read_jsonl(args.a), read_jsonl(args.b), args.seed)
        with args.out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]))
            w.writeheader()
            w.writerows(sheet)
        args.out.with_suffix(".key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
        print(f"wrote {len(sheet)} blinded pairs to {args.out} (key: {args.out.with_suffix('.key.json')})")
    elif args.cmd == "agreement":
        with args.sheet.open(encoding="utf-8") as f:
            sheet = list(csv.DictReader(f))
        key = json.loads(args.key.read_text(encoding="utf-8"))
        judge = {r["id"]: r["winner"] for r in read_jsonl(args.judge)}
        print(json.dumps(agreement(sheet, key, judge), indent=2))


if __name__ == "__main__":
    main()
