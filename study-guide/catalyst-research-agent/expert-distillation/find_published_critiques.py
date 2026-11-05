r"""Find published critiques (Comment / Reply exchanges, retractions) on your themes via OpenAlex.

These form the independent expert test set (expert-distillation.md, section 4.6): real experts
already critiqued a specific paper in public, so we can test whether the agent raises the same
issues, without asking anyone for their time.

    python find_published_critiques.py search --out critiques/ \
        --terms "zeolite" "zeolite membrane" "MFI" "nanosheet" "molecular sieve" --from-year 1995
    python find_published_critiques.py selftest

Outputs in <out>/:
    critiques.jsonl     one record per (critique -> target paper) link, with abstracts, match score
                        and an empty "issues" list for you (or critique_eval.py extract-issues) to fill
    unmatched.jsonl     critique-like titles whose target paper could not be identified (check by hand)
    retractions.jsonl   retracted works on the same themes (optional, --retractions)

Environment variables: OPENALEX_MAILTO, OPENALEX_API_KEY (optional, as in build_expert_corpus.py).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling modules, even under python -I

from build_expert_corpus import (Fetcher, abstract_from_inverted_index, clean_doi, openalex_fetcher,
                                 short_id, venue_name)

FIELDS = ",".join(["id", "doi", "display_name", "publication_year", "type", "primary_location",
                   "abstract_inverted_index", "referenced_works", "authorships", "cited_by_count"])

CRITIQUE_TITLE = re.compile(
    r"^\s*(?P<kind>comments?\s+on|reply\s+to|response\s+to|rebuttal\s+to|"
    r"correspondence\s*(on|regarding)?|matters\s+arising|discussion\s+of)\b[\s:]*(?P<rest>.*)$", re.I)


def critique_kind(title: str) -> tuple[str, str] | None:
    """('comment'|'reply', remainder of title) for critique-like titles, else None."""
    m = CRITIQUE_TITLE.match(title or "")
    if not m:
        return None
    word = m.group("kind").lower()
    kind = "reply" if word.startswith(("reply", "response", "rebuttal")) else "comment"
    return kind, m.group("rest").strip()


def target_title_hint(rest: str) -> str:
    """Target title from 'Comment on "X" [J. Abc 12 (2020) 3]', including nested 'Reply to "Comment on X"'."""
    hint = rest
    while True:
        hint = re.sub(r"[\"“”‘’]", "", hint)
        hint = re.sub(r"\s*\[[^\]]*\]\s*$", "", hint)  # trailing [J. Abc 12 (2020) 3]
        hint = re.sub(r"\s*\([^)]*\d{4}[^)]*\)\s*$", "", hint)  # trailing (J. Abc, 2020)
        nested = critique_kind(hint)
        if not nested:
            return hint.strip(" .:'")
        hint = nested[1]


def similarity(a: str, b: str) -> float:
    norm = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, " ".join(norm(a).split()), " ".join(norm(b).split())).ratio()


def search_critiques(fetch: Fetcher, terms: list[str], from_year: int, max_per_query: int
                     ) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for term in terms:
        for prefix in ("comment on", "reply to", "response to", "matters arising"):
            cursor, n = "*", 0
            while cursor and n < max_per_query:
                res = fetch("/works", {
                    "filter": f"title.search:{prefix},from_publication_date:{from_year}-01-01",
                    "search": term, "per-page": 100, "cursor": cursor, "select": FIELDS})
                for w in res.get("results", []):
                    n += 1
                    if critique_kind(w.get("display_name", "")):
                        found[short_id(w["id"])] = w
                cursor = (res.get("meta") or {}).get("next_cursor")
    return found


def fetch_works(fetch: Fetcher, ids: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        res = fetch("/works", {"filter": "openalex:" + "|".join(chunk), "per-page": 50, "select": FIELDS})
        out.update({short_id(w["id"]): w for w in res.get("results", [])})
    return out


def link_targets(fetch: Fetcher, critiques: dict[str, dict[str, Any]], min_score: float
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ref_ids = sorted({short_id(r) for w in critiques.values() for r in w.get("referenced_works") or []})
    refs = fetch_works(fetch, ref_ids)
    linked, unmatched = [], []
    for cid, w in critiques.items():
        kind, rest = critique_kind(w["display_name"])
        hint = target_title_hint(rest)
        best, best_score = None, 0.0
        for r in w.get("referenced_works") or []:
            ref = refs.get(short_id(r))
            if ref and not critique_kind(ref.get("display_name", "")):
                score = similarity(hint, ref.get("display_name", ""))
                if score > best_score:
                    best, best_score = ref, score
        base = {"critique_id": cid, "kind": kind, "critique_doi": clean_doi(w.get("doi")),
                "critique_title": w["display_name"], "critique_year": w.get("publication_year"),
                "critique_venue": venue_name(w),
                "critique_abstract": abstract_from_inverted_index(w.get("abstract_inverted_index")),
                "target_title_hint": hint}
        if best and best_score >= min_score:
            linked.append({**base, "target_id": short_id(best["id"]), "target_doi": clean_doi(best.get("doi")),
                           "target_title": best["display_name"], "target_year": best.get("publication_year"),
                           "target_abstract": abstract_from_inverted_index(best.get("abstract_inverted_index")),
                           "match_score": round(best_score, 3), "issues": [], "checked": False})
        else:
            unmatched.append({**base, "best_score": round(best_score, 3)})
    # attach replies to the comment on the same target, so you can see both sides
    by_target: dict[str, list[dict[str, Any]]] = {}
    for rec in linked:
        by_target.setdefault(rec["target_id"], []).append(rec)
    for recs in by_target.values():
        ids = [r["critique_id"] for r in recs]
        for r in recs:
            r["related_critiques"] = [i for i in ids if i != r["critique_id"]]
    return linked, unmatched


def search_retractions(fetch: Fetcher, terms: list[str], from_year: int) -> list[dict[str, Any]]:
    out = {}
    for term in terms:
        res = fetch("/works", {"filter": f"is_retracted:true,from_publication_date:{from_year}-01-01",
                               "search": term, "per-page": 100, "select": FIELDS})
        for w in res.get("results", []):
            out[short_id(w["id"])] = {"id": short_id(w["id"]), "doi": clean_doi(w.get("doi")),
                                      "title": w.get("display_name"), "year": w.get("publication_year"),
                                      "venue": venue_name(w)}
    return list(out.values())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------

def _fixture_fetcher() -> Fetcher:
    def w(i, title, refs=(), abstract=None):
        return {"id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/10.0000/x.{i}", "display_name": title,
                "publication_year": 2020, "type": "article", "primary_location": {"source": {"display_name": "J"}},
                "abstract_inverted_index": abstract, "referenced_works": [f"https://openalex.org/W{r}" for r in refs]}
    target = w(1, "Ultrathin example zeolite nanosheet membranes for isomer separation",
               abstract={"We": [0], "report": [1], "membranes.": [2]})
    other = w(2, "Unrelated adsorption study")
    comment = w(10, 'Comment on "Ultrathin Example Zeolite Nanosheet Membranes for Isomer Separation"', refs=(1, 2),
                abstract={"Thickness": [0], "was": [1], "not": [2], "measured.": [3]})
    reply = w(11, 'Reply to "Comment on “Ultrathin example zeolite nanosheet membranes for isomer separation”"',
              refs=(1, 10))
    orphan = w(12, "Comment on a paper we cannot find", refs=(2,))
    noise = w(13, "Commentary: zeolites in 2030")

    def fetch(path, params=None):
        f = params.get("filter", "")
        if f.startswith("openalex:"):
            ids = f.split(":", 1)[1].split("|")
            pool = {x["id"].rsplit("/", 1)[-1]: x for x in (target, other, comment, reply, orphan)}
            return {"results": [pool[i] for i in ids if i in pool]}
        if "is_retracted" in f:
            return {"results": [w(20, "Retracted example")]}
        return {"meta": {"next_cursor": None}, "results": [comment, reply, orphan, noise]}
    return fetch


def selftest() -> None:
    assert critique_kind('Comment on "X"')[0] == "comment"
    assert critique_kind("Reply to Comment on X")[0] == "reply"
    assert critique_kind("Commentary: zeolites") is None
    assert target_title_hint('Reply to "Comment on “Abc def”"') == "Abc def"
    fetch = _fixture_fetcher()
    found = search_critiques(fetch, ["zeolite"], 1995, 100)
    assert set(found) == {"W10", "W11", "W12"}, found
    linked, unmatched = link_targets(fetch, found, 0.75)
    assert {r["critique_id"] for r in linked} == {"W10", "W11"}, linked
    assert all(r["target_id"] == "W1" for r in linked)
    assert linked[0]["related_critiques"] and [u["critique_id"] for u in unmatched] == ["W12"]
    assert search_retractions(fetch, ["zeolite"], 1995)[0]["id"] == "W20"
    print("selftest passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--terms", nargs="+", default=["zeolite", "zeolite membrane", "molecular sieve", "nanosheet"])
    s.add_argument("--from-year", type=int, default=1995)
    s.add_argument("--max-per-query", type=int, default=500)
    s.add_argument("--min-match", type=float, default=0.75, help="title similarity needed to link a target")
    s.add_argument("--retractions", action="store_true")
    sub.add_parser("selftest")
    args = p.parse_args()
    if args.cmd == "selftest":
        selftest()
        return
    fetch = openalex_fetcher()
    found = search_critiques(fetch, args.terms, args.from_year, args.max_per_query)
    linked, unmatched = link_targets(fetch, found, args.min_match)
    write_jsonl(args.out / "critiques.jsonl", linked)
    write_jsonl(args.out / "unmatched.jsonl", unmatched)
    print(f"{len(found)} critique-like works; {len(linked)} linked to a target, {len(unmatched)} unmatched")
    if args.retractions:
        rows = search_retractions(fetch, args.terms, args.from_year)
        write_jsonl(args.out / "retractions.jsonl", rows)
        print(f"{len(rows)} retracted works")


if __name__ == "__main__":
    main()
