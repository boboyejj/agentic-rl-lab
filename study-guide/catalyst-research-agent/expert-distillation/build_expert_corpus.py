r"""Build an expert corpus (publications, reviews, perspectives) from author names via OpenAlex.

Standard library only. Runs on a laptop CPU.

Three subcommands, run in order:

    # 1. Find OpenAlex author profiles for each expert in the config.
    #    Prints candidates; writes <out>/author_candidates.json.
    python build_expert_corpus.py resolve --config experts.example.toml --out corpus/

    # 2. Copy the right OpenAlex IDs (and/or ORCID) into the config, then harvest works.
    python build_expert_corpus.py harvest --config experts.toml --out corpus/

    # 3. (optional) Add Semantic Scholar TLDRs / influential-citation counts.
    python build_expert_corpus.py enrich --out corpus/

    # Offline check of the parsing/classification logic (no network):
    python build_expert_corpus.py selftest

Outputs in <out>/:
    author_candidates.json   resolve: candidate profiles per expert, for you to confirm
    works.jsonl              one record per unique work (deduplicated by DOI / OpenAlex ID)
    opinion_docs.jsonl       reviews, perspectives, accounts, theses: the rubric-extraction queue
    summary.csv              per-expert counts by document role
    fetch_queue.csv          open-access PDF links + licences, for a separate, polite download step

Environment variables (all optional):
    OPENALEX_MAILTO   contact address for OpenAlex's polite pool (use a lab/project address)
    OPENALEX_API_KEY  OpenAlex API key, if your account has one
    S2_API_KEY        Semantic Scholar API key (higher rate limit for `enrich`)

This script only collects metadata, abstracts and OA links. It does not download full
texts. Downloading and storing full texts must follow each publisher's licence or your
library's text-and-data-mining (TDM) agreement (see zeolite-plan.md, section 4).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

OPENALEX = "https://api.openalex.org"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
USER_AGENT = "agentic-rl-lab-expert-corpus/0.1"

WORK_FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "publication_date", "type",
    "primary_location", "best_oa_location", "open_access", "authorships",
    "abstract_inverted_index", "cited_by_count", "primary_topic", "language",
])

# --------------------------------------------------------------------------------------
# Document roles. What a work is *for* in the distillation pipeline:
#   opinion  -> reviews, perspectives, accounts, viewpoints: source of rubrics and critique style
#   thesis   -> dissertations from the group: methods detail, "lessons learned"
#   primary  -> research articles: factual QA, extraction, eval questions
#   other    -> errata, editorials without content, datasets, etc.
# --------------------------------------------------------------------------------------

OPINION_TITLE_PATTERNS = [
    r"\bperspective", r"\boutlook\b", r"\bviewpoint\b", r"\bcommentary\b", r"\bopinion\b",
    r"\breview\b", r"\boverview\b", r"\bprogress in\b", r"\badvances in\b", r"\brecent advances\b",
    r"\bchallenges\b", r"\bopportunities\b", r"\broadmap\b", r"\bprospects\b", r"\bfuture of\b",
    r"\btutorial\b", r"\blessons\b", r"\bstate of the art\b", r"\bwhere (are|do) we\b",
    r"\bquo vadis\b", r"\bcritical assessment\b",
]
OPINION_VENUE_PATTERNS = [
    r"accounts of chemical research", r"chemical reviews", r"chemical society reviews",
    r"annual review", r"nature reviews", r"current opinion", r"trends in",
    r"progress in materials science", r"catalysis reviews", r"perspective",
]
NON_CONTENT_TITLE = re.compile(r"^(erratum|correction|corrigendum|retraction|editorial|front cover|"
                               r"back cover|inside cover|cover picture|frontispiece)\b", re.I)


def classify_role(work: dict[str, Any]) -> str:
    title = (work.get("display_name") or "").strip()
    wtype = (work.get("type") or "").lower()
    venue = venue_name(work).lower()
    if NON_CONTENT_TITLE.search(title) or wtype in {"erratum", "paratext", "retraction", "dataset"}:
        return "other"
    if wtype == "dissertation":
        return "thesis"
    if wtype == "review":
        return "opinion"
    if any(re.search(p, title, re.I) for p in OPINION_TITLE_PATTERNS):
        return "opinion"
    if any(re.search(p, venue) for p in OPINION_VENUE_PATTERNS):
        return "opinion"
    if wtype in {"article", "preprint", "letter", "book-chapter", "proceedings-article"}:
        return "primary"
    return "other"


# --------------------------------------------------------------------------------------
# Small helpers over OpenAlex records
# --------------------------------------------------------------------------------------

def venue_name(work: dict[str, Any]) -> str:
    source = ((work.get("primary_location") or {}).get("source") or {})
    return source.get("display_name") or ""


def short_id(openalex_id: str | None) -> str:
    return (openalex_id or "").rstrip("/").rsplit("/", 1)[-1]


def clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I).lower()


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not index:
        return ""
    slots: dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def author_position(work: dict[str, Any], author_ids: set[str]) -> str | None:
    """'first', 'middle' or 'last' for the expert on this work (None if not found)."""
    for authorship in work.get("authorships") or []:
        if short_id((authorship.get("author") or {}).get("id")) in author_ids:
            return authorship.get("author_position")
    return None


def matches_terms(text: str, terms: Iterable[str]) -> bool:
    terms = [t.lower() for t in terms if t]
    return not terms or any(t in text.lower() for t in terms)


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------

Fetcher = Callable[[str, dict[str, Any] | None], dict[str, Any]]


def http_json(url: str, body: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None,
              retries: int = 5) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def openalex_fetcher() -> Fetcher:
    extra = {}
    if os.getenv("OPENALEX_MAILTO"):
        extra["mailto"] = os.environ["OPENALEX_MAILTO"]
    if os.getenv("OPENALEX_API_KEY"):
        extra["api_key"] = os.environ["OPENALEX_API_KEY"]

    def fetch(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode({**(params or {}), **extra})
        time.sleep(0.12)  # stay well under OpenAlex's ~10 requests/second
        return http_json(f"{OPENALEX}{path}?{query}")

    return fetch


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class Expert:
    key: str
    name: str
    openalex_ids: list[str] = field(default_factory=list)
    orcid: str = ""
    institution_hint: str = ""
    themes: list[str] = field(default_factory=list)
    topic_terms: list[str] = field(default_factory=list)
    consent: str = "published-only"  # published-only | participating
    from_year: int | None = None


def load_config(path: Path) -> tuple[dict[str, Any], list[Expert]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    settings = raw.get("settings", {})
    known = set(Expert.__dataclass_fields__)
    experts = []
    for item in raw.get("expert", []):
        unknown = set(item) - known
        if unknown:
            raise ValueError(f"expert {item.get('key')}: unknown fields {sorted(unknown)}")
        experts.append(Expert(**item))
    keys = [e.key for e in experts]
    if len(keys) != len(set(keys)):
        raise ValueError("expert keys must be unique")
    return settings, experts


# --------------------------------------------------------------------------------------
# resolve: names -> candidate OpenAlex author profiles
# --------------------------------------------------------------------------------------

def summarize_author(a: dict[str, Any]) -> dict[str, Any]:
    insts = [i.get("display_name", "") for i in a.get("last_known_institutions") or []]
    past = sorted({(aff.get("institution") or {}).get("display_name", "")
                   for aff in a.get("affiliations") or []} - set(insts) - {""})
    topics = [t.get("display_name", "") for t in (a.get("topics") or [])[:5]]
    return {
        "openalex_id": short_id(a.get("id")), "display_name": a.get("display_name"),
        "orcid": a.get("orcid"), "works_count": a.get("works_count"),
        "cited_by_count": a.get("cited_by_count"),
        "h_index": (a.get("summary_stats") or {}).get("h_index"),
        "last_known_institutions": insts, "past_institutions": past[:6], "top_topics": topics,
    }


def score_candidate(c: dict[str, Any], expert: Expert) -> float:
    """Heuristic ranking only. A human confirms the final choice."""
    score = 0.0
    hint = expert.institution_hint.lower()
    if hint and any(hint in i.lower() for i in c["last_known_institutions"] + c["past_institutions"]):
        score += 3
    terms = [t.lower() for t in expert.themes + expert.topic_terms]
    topics = " ".join(c["top_topics"]).lower()
    score += sum(1 for t in terms if any(w in topics for w in t.split() if len(w) > 4))
    score += min((c["works_count"] or 0) / 200, 2)
    return score


def resolve(fetch: Fetcher, experts: list[Expert]) -> dict[str, Any]:
    report = {}
    for e in experts:
        if e.orcid:
            orcid = e.orcid if e.orcid.startswith("http") else f"https://orcid.org/{e.orcid}"
            try:
                cands = [summarize_author(fetch(f"/authors/{orcid}", None))]
            except urllib.error.HTTPError:
                cands = []
        else:
            cands = []
        if not cands:
            res = fetch("/authors", {"search": e.name, "per-page": 15})
            cands = [summarize_author(a) for a in res.get("results", [])]
        for c in cands:
            c["score"] = round(score_candidate(c, e), 2)
        cands.sort(key=lambda c: -c["score"])
        report[e.key] = {"name": e.name, "institution_hint": e.institution_hint, "candidates": cands}
    return report


def print_resolve_report(report: dict[str, Any]) -> None:
    for key, item in report.items():
        print(f"\n== {key}: {item['name']} (hint: {item['institution_hint'] or '-'})")
        for c in item["candidates"][:6]:
            inst = "; ".join(c["last_known_institutions"]) or "?"
            print(f"  {c['openalex_id']:<12} score={c['score']:<5} works={c['works_count']:<5} "
                  f"h={c['h_index']}  {c['display_name']} | {inst} | {', '.join(c['top_topics'][:3])}")
    print("\nOpenAlex sometimes splits one person into several profiles. Put *all* correct IDs into "
          "openalex_ids in your config (or set orcid), then run `harvest`.")


# --------------------------------------------------------------------------------------
# harvest: confirmed author IDs -> works
# --------------------------------------------------------------------------------------

def iter_author_works(fetch: Fetcher, author_ids: list[str], from_year: int | None,
                      max_works: int) -> Iterable[dict[str, Any]]:
    filt = "author.id:" + "|".join(author_ids)
    if from_year:
        filt += f",from_publication_date:{from_year}-01-01"
    cursor, seen = "*", 0
    while cursor and seen < max_works:
        res = fetch("/works", {"filter": filt, "per-page": 200, "cursor": cursor, "select": WORK_FIELDS})
        for work in res.get("results", []):
            yield work
            seen += 1
            if seen >= max_works:
                return
        cursor = (res.get("meta") or {}).get("next_cursor")


def make_record(work: dict[str, Any], expert: Expert) -> dict[str, Any]:
    ids = {short_id(i) for i in expert.openalex_ids}
    oa = work.get("open_access") or {}
    best = work.get("best_oa_location") or {}
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))
    title = work.get("display_name") or ""
    authors = [(a.get("author") or {}).get("display_name") for a in work.get("authorships") or []]
    return {
        "openalex_id": short_id(work.get("id")),
        "doi": clean_doi(work.get("doi")),
        "title": title,
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "role": classify_role(work),
        "venue": venue_name(work),
        "authors": authors,
        "abstract": abstract,
        "cited_by_count": work.get("cited_by_count"),
        "topic": ((work.get("primary_topic") or {}).get("display_name")),
        "is_oa": bool(oa.get("is_oa")),
        "oa_status": oa.get("oa_status"),
        "oa_pdf_url": best.get("pdf_url"),
        "oa_landing_url": best.get("landing_page_url") or oa.get("oa_url"),
        "license": best.get("license"),
        "experts": [expert.key],
        "expert_positions": {expert.key: author_position(work, ids)},
        "on_topic": matches_terms(f"{title} {abstract}", expert.topic_terms),
        "consent": {expert.key: expert.consent},
    }


def merge_record(into: dict[str, Any], new: dict[str, Any]) -> None:
    for key in new["experts"]:
        if key not in into["experts"]:
            into["experts"].append(key)
    into["expert_positions"].update(new["expert_positions"])
    into["consent"].update(new["consent"])
    into["on_topic"] = into["on_topic"] or new["on_topic"]


def harvest(fetch: Fetcher, settings: dict[str, Any], experts: list[Expert],
            max_works: int) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    for e in experts:
        if not e.openalex_ids:
            print(f"[skip] {e.key}: no openalex_ids in config (run `resolve` first)", file=sys.stderr)
            continue
        ids = [short_id(i) for i in e.openalex_ids]
        from_year = e.from_year or settings.get("from_year")
        n = 0
        for work in iter_author_works(fetch, ids, from_year, max_works):
            if work.get("language") not in (None, "en"):
                continue
            rec = make_record(work, e)
            key = rec["doi"] or rec["openalex_id"]
            if key in corpus:
                merge_record(corpus[key], rec)
            else:
                corpus[key] = rec
            n += 1
        print(f"[ok] {e.key}: {n} works")
    return corpus


def write_outputs(corpus: dict[str, dict[str, Any]], experts: list[Expert], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records = sorted(corpus.values(), key=lambda r: (-(r["year"] or 0), r["title"]))
    with (out / "works.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "opinion_docs.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            if r["role"] in {"opinion", "thesis"} and r["on_topic"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["expert", "works", "primary", "opinion", "thesis", "other", "off_topic",
                    "open_access", "senior_author(last)", "first_author"])
        for e in experts:
            mine = [r for r in records if e.key in r["experts"]]
            count = lambda role: sum(r["role"] == role for r in mine)  # noqa: E731
            w.writerow([e.key, len(mine), count("primary"), count("opinion"), count("thesis"),
                        count("other"), sum(not r["on_topic"] for r in mine),
                        sum(r["is_oa"] for r in mine),
                        sum(r["expert_positions"].get(e.key) == "last" for r in mine),
                        sum(r["expert_positions"].get(e.key) == "first" for r in mine)])
    with (out / "fetch_queue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doi", "openalex_id", "role", "license", "oa_status", "oa_pdf_url", "oa_landing_url"])
        for r in records:
            if r["is_oa"] and r["role"] != "other":
                w.writerow([r["doi"], r["openalex_id"], r["role"], r["license"], r["oa_status"],
                            r["oa_pdf_url"], r["oa_landing_url"]])
    print(f"wrote {len(records)} works to {out}/works.jsonl "
          f"({sum(r['role'] == 'opinion' for r in records)} opinion docs)")


# --------------------------------------------------------------------------------------
# enrich: Semantic Scholar TLDRs and influential citations
# --------------------------------------------------------------------------------------

def enrich(out: Path) -> None:
    path = out / "works.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.getenv("S2_API_KEY") else {}
    with_doi = [r for r in records if r.get("doi")]
    fields = "tldr,citationCount,influentialCitationCount,publicationTypes,externalIds"
    for start in range(0, len(with_doi), 400):
        chunk = with_doi[start:start + 400]
        res = http_json(f"{S2_BATCH}?fields={fields}", {"ids": [f"DOI:{r['doi']}" for r in chunk]},
                        headers=headers)
        for rec, paper in zip(chunk, res if isinstance(res, list) else []):
            if not paper:
                continue
            rec["s2_tldr"] = (paper.get("tldr") or {}).get("text")
            rec["s2_influential_citations"] = paper.get("influentialCitationCount")
            rec["s2_publication_types"] = paper.get("publicationTypes")
            if "Review" in (paper.get("publicationTypes") or []) and rec["role"] == "primary":
                rec["role"] = "opinion"
        time.sleep(1.1 if not headers else 0.2)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"enriched {len(with_doi)} works with DOIs")


# --------------------------------------------------------------------------------------
# selftest: offline check with tiny synthetic OpenAlex-shaped fixtures (no real data)
# --------------------------------------------------------------------------------------

def _fixture_fetcher() -> Fetcher:
    authors = {"results": [
        {"id": "https://openalex.org/A1", "display_name": "Expert One", "orcid": None, "works_count": 300,
         "cited_by_count": 9000, "summary_stats": {"h_index": 50},
         "last_known_institutions": [{"display_name": "Example University"}],
         "affiliations": [], "topics": [{"display_name": "Zeolite Synthesis and Applications"}]},
        {"id": "https://openalex.org/A2", "display_name": "Expert One", "orcid": None, "works_count": 12,
         "cited_by_count": 40, "summary_stats": {"h_index": 3},
         "last_known_institutions": [{"display_name": "Other Institute"}],
         "affiliations": [], "topics": [{"display_name": "Plant Biology"}]},
    ]}

    def work(i: int, title: str, wtype: str, venue: str, pos: str, oa: bool) -> dict[str, Any]:
        return {
            "id": f"https://openalex.org/W{i}", "doi": f"https://doi.org/10.0000/example.{i}",
            "display_name": title, "publication_year": 2020 + i % 4, "type": wtype, "language": "en",
            "primary_location": {"source": {"display_name": venue}},
            "best_oa_location": {"pdf_url": f"https://example.org/{i}.pdf" if oa else None,
                                 "license": "cc-by" if oa else None},
            "open_access": {"is_oa": oa, "oa_status": "gold" if oa else "closed"},
            "authorships": [{"author": {"id": "https://openalex.org/A9", "display_name": "Co Author"},
                             "author_position": "first"},
                            {"author": {"id": "https://openalex.org/A1", "display_name": "Expert One"},
                             "author_position": pos}],
            "abstract_inverted_index": {"Zeolite": [0], "nanosheets": [1], "were": [2], "made.": [3]},
            "cited_by_count": 10 * i, "primary_topic": {"display_name": "Zeolite Synthesis"},
        }

    works = {"meta": {"next_cursor": None}, "results": [
        work(1, "Direct synthesis of example nanosheets", "article", "Example Journal", "last", True),
        work(2, "Perspective: challenges in two-dimensional zeolites", "article", "Example J", "last", True),
        work(3, "Membranes for isomer separation", "review", "Example Reviews", "first", False),
        work(4, "Erratum: Direct synthesis of example nanosheets", "erratum", "Example Journal", "last", False),
        work(5, "Advances in zeolite membranes", "article", "Accounts of Chemical Research", "last", False),
    ]}

    def fetch(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return authors if path == "/authors" else works

    return fetch


def selftest() -> None:
    import tempfile
    fetch = _fixture_fetcher()
    e = Expert(key="one", name="Expert One", institution_hint="Example University",
               themes=["zeolite synthesis"], topic_terms=["zeolite", "membrane"])
    rep = resolve(fetch, [e])
    assert rep["one"]["candidates"][0]["openalex_id"] == "A1", rep
    e.openalex_ids = ["A1"]
    corpus = harvest(fetch, {}, [e], max_works=100)
    roles = {r["openalex_id"]: r["role"] for r in corpus.values()}
    assert roles == {"W1": "primary", "W2": "opinion", "W3": "opinion", "W4": "other", "W5": "opinion"}, roles
    rec = corpus["10.0000/example.1"]
    assert rec["abstract"] == "Zeolite nanosheets were made." and rec["expert_positions"]["one"] == "last"
    with tempfile.TemporaryDirectory() as tmp:
        write_outputs(corpus, [e], Path(tmp))
        n_op = sum(1 for _ in open(Path(tmp) / "opinion_docs.jsonl"))
        assert n_op == 3, n_op
    print("selftest passed")


# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("resolve", "harvest"):
        s = sub.add_parser(name)
        s.add_argument("--config", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        if name == "harvest":
            s.add_argument("--max-works-per-expert", type=int, default=2000)
    s = sub.add_parser("enrich")
    s.add_argument("--out", type=Path, required=True)
    sub.add_parser("selftest")
    args = p.parse_args()

    if args.cmd == "selftest":
        selftest()
        return
    if args.cmd == "enrich":
        enrich(args.out)
        return
    settings, experts = load_config(args.config)
    fetch = openalex_fetcher()
    if args.cmd == "resolve":
        report = resolve(fetch, experts)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "author_candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_resolve_report(report)
    else:
        corpus = harvest(fetch, settings, experts, args.max_works_per_expert)
        write_outputs(corpus, experts, args.out)


if __name__ == "__main__":
    main()
