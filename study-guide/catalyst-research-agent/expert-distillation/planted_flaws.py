r"""Mini-proposals with known ("planted") flaws, for scoring critiques automatically.

Each mini-proposal is built from a published paper (aim from its title, background from the
first sentences of its abstract, results withheld) plus an "approach" paragraph. For every rubric
dimension the approach uses either a sound sentence or, if a flaw is planted there, a flawed one.
A good critique must find the planted flaws (recall) and should not invent them in clean
controls (false alarms). See expert-distillation.md, section 5.1.

    python planted_flaws.py list                       # print the flaw library
    python planted_flaws.py make --works corpus/works.jsonl --out planted.jsonl --n 60
    python planted_flaws.py selftest

Edit FLAWS to add your own: the library is the main thing you write by hand. Each flaw has a
plain-language description (used by the LLM judge) and regex `detect` groups (a cheap first-pass
check: every group must match somewhere in the critique).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Flaw:
    id: str
    dimension: str
    contexts: tuple[str, ...]          # synthesis | membrane | catalysis
    flawed: str                         # sentence inserted into the approach
    description: str                    # what is wrong (for the judge and for you)
    detect: tuple[str, ...] = field(default_factory=tuple)  # regex groups, all must match


# Sound sentence per (dimension, context). Used when no flaw is planted in that dimension.
SOUND: dict[tuple[str, str], str] = {
    ("characterization_evidence", "synthesis"): "Products will be characterized by XRD, SEM/TEM, AFM thickness statistics on at least 50 sheets, Ar physisorption and 29Si/27Al NMR.",
    ("characterization_evidence", "membrane"): "Membrane microstructure will be characterized by cross-sectional SEM, TEM of the selective layer and XRD of the supported film.",
    ("characterization_evidence", "catalysis"): "Acid-site density and type will be quantified by NH3-TPD and pyridine FTIR, and crystal size by SEM/TEM.",
    ("synthesis_reproducibility", "synthesis"): "Full molar gel compositions, reagent sources, aging and crystallization conditions will be reported, and each condition repeated in three independent batches.",
    ("synthesis_reproducibility", "membrane"): "Each membrane recipe will be repeated on at least five supports to report batch-to-batch variability.",
    ("synthesis_reproducibility", "catalysis"): "Catalysts will be prepared in three batches and tested in duplicate.",
    ("mechanism_claims", "synthesis"): "Crystallization will be followed ex situ by XRD, TEM and NMR at intermediate times to test competing formation mechanisms.",
    ("structure_property_link", "membrane"): "A series of membranes with systematically varied thickness and defect density will be used to relate structure to permeance and selectivity.",
    ("structure_property_link", "catalysis"): "A series of samples with systematically varied crystal thickness at constant Si/Al will separate diffusion effects from acid-site effects.",
    ("performance_benchmarking", "membrane"): "Binary-mixture separation factors will be measured and compared with literature values at matched temperature, feed partial pressures and permeate conditions, over at least 100 h.",
    ("performance_benchmarking", "catalysis"): "Activity will be compared with a commercial reference zeolite under identical conditions, with conversion kept below 15 % for rate measurements and stability tested over time on stream.",
    ("scalability_cost", "synthesis"): "OSDA cost per kg of product and options for OSDA-free or recycled-OSDA routes will be estimated.",
    ("scalability_cost", "membrane"): "Scale-up will be assessed on tubular supports with at least 10 cm2 area, and long-term stability under humid feed will be tested.",
    ("novelty_positioning", "synthesis"): "The closest prior syntheses will be compared in a table of conditions and outcomes to make the contribution explicit.",
    ("novelty_positioning", "membrane"): "The work will be positioned against the best reported membranes for this separation in a selectivity-permeance plot.",
    ("novelty_positioning", "catalysis"): "The study will be positioned against prior hierarchical and nanosheet catalysts for the same reaction.",
}

FLAWS: list[Flaw] = [
    Flaw("thickness_xrd_only", "characterization_evidence", ("synthesis", "catalysis"),
         "Nanosheet thickness will be determined from the broadening of the XRD reflections using the Scherrer equation.",
         "Thickness is inferred only from XRD peak broadening; it needs direct measurement (AFM/TEM statistics), "
         "since broadening also reflects strain, defects and instrument contributions.",
         (r"thickness|scherrer|broadening", r"\bAFM\b|\bTEM\b|microscop|direct(ly)? measur")),
    Flaw("single_image_morphology", "characterization_evidence", ("synthesis", "membrane", "catalysis"),
         "The morphology will be confirmed by a representative SEM image of the product.",
         "Morphology is claimed from a single representative image; it needs statistics over many particles/areas.",
         (r"representative|single|one (sem )?image|statistic", r"\bSEM\b|image|morpholog")),
    Flaw("no_porosity", "characterization_evidence", ("synthesis", "catalysis"),
         "Phase purity will be confirmed by XRD; no further characterization of porosity is planned.",
         "Microporosity is never measured (no Ar/N2 physisorption, micropore volume); XRD alone cannot show accessible pores.",
         (r"physisorption|isotherm|micropore|\bBET\b|porosity|pore volume",)),
    Flaw("single_batch", "synthesis_reproducibility", ("synthesis", "membrane", "catalysis"),
         "All conclusions will be based on one synthesis batch for each condition.",
         "Single batches give no measure of reproducibility; zeolite syntheses are notoriously batch-sensitive.",
         (r"batch|reproducib|repeat|replicat",)),
    Flaw("vague_gel", "synthesis_reproducibility", ("synthesis",),
         "The synthesis gel will be prepared from TEOS and an organic structure-directing agent following our usual procedure.",
         "Gel composition, reagent amounts and aging are not specified, so the synthesis cannot be reproduced or compared.",
         (r"gel|composition|molar|ratio|reproduc|aging|ageing",)),
    Flaw("mechanism_from_endpoint", "mechanism_claims", ("synthesis",),
         "The formation mechanism will be inferred from the structure of the final crystalline product.",
         "A formation mechanism cannot be inferred from the end product alone; intermediate-time (or in situ) data are needed.",
         (r"mechanism|formation|growth", r"intermediate|time-resolved|in situ|kinetic|over time|final product|end ?product")),
    Flaw("two_point_correlation", "structure_property_link", ("membrane", "catalysis"),
         "The effect of nanosheet morphology will be shown by comparing one nanosheet sample with one conventional sample.",
         "A two-sample comparison confounds morphology with Si/Al, defects and crystallinity; a systematic series is needed.",
         (r"two samples|one (nanosheet|conventional) sample|single comparison|confound|systematic|series|control",)),
    Flaw("mismatched_benchmark", "performance_benchmarking", ("membrane",),
         "Separation factors will be compared with the best literature values regardless of the temperature and feed conditions used.",
         "Benchmarking against literature values measured at different temperature/feed pressures is not a fair comparison.",
         (r"compar|benchmark|literature", r"condition|temperature|pressure|feed|matched|same")),
    Flaw("ideal_selectivity_only", "performance_benchmarking", ("membrane",),
         "Membrane selectivity will be determined as the ratio of single-component permeances.",
         "Ideal (single-gas) selectivity can differ strongly from mixture separation factors because of competitive adsorption.",
         (r"single[- ](gas|component)|ideal selectivity|mixture|binary|competitive",)),
    Flaw("short_stability", "performance_benchmarking", ("membrane", "catalysis"),
         "Performance will be evaluated during the first two hours of operation.",
         "Two hours says nothing about stability (defect formation, coking, deactivation); long-term tests are needed.",
         (r"stabilit|long[- ]term|deactivat|time on stream|hours|durab",)),
    Flaw("high_conversion_rates", "performance_benchmarking", ("catalysis",),
         "Intrinsic rates will be compared at 90 % conversion.",
         "Rates at high conversion are not differential and may be transport- or equilibrium-limited; compare at low conversion.",
         (r"conversion", r"differential|low conversion|transport|diffusion|equilibri|kinetic regime")),
    Flaw("osda_cost_ignored", "scalability_cost", ("synthesis", "membrane"),
         "A custom multi-step organic structure-directing agent will be used at OSDA/SiO2 = 0.5, without considering cost or recovery.",
         "High loading of a costly custom OSDA is not discussed, which undermines the claimed practicality.",
         (r"cost|expens|recycl|recover|scal|OSDA|template", r"cost|expens|recycl|recover|scal|practical")),
    Flaw("tiny_area_extrapolation", "scalability_cost", ("membrane",),
         "Results from 0.5 cm2 membrane discs will be extrapolated directly to industrial module performance.",
         "Defect density scales with area; tiny discs cannot support module-level claims.",
         (r"area|scale|module|extrapolat|defect",)),
    Flaw("first_without_prior_art", "novelty_positioning", ("synthesis", "membrane", "catalysis"),
         "This will be the first report of this type of material, so no comparison with prior work is needed.",
         "Claims of being 'first' without a prior-art comparison; closely related routes likely exist.",
         (r"first|prior (art|work)|previous|novel|literature|compar",)),
]
FLAW_BY_ID = {f.id: f for f in FLAWS}
assert all(isinstance(f.detect, tuple) and f.detect for f in FLAWS), "detect must be a non-empty tuple (trailing comma!)"
assert len(FLAW_BY_ID) == len(FLAWS), "duplicate flaw ids"

CONTEXT_TERMS = {
    "membrane": r"membrane|permea|separation|film",
    "catalysis": r"catal|conversion|reaction|cracking|alkylation|isomeri|methanol",
}


def infer_context(text: str) -> str:
    for ctx, pattern in CONTEXT_TERMS.items():
        if re.search(pattern, text, re.I):
            return ctx
    return "synthesis"


RESULT_START = re.compile(r"^(here(in)?|we |in this (work|study|paper|contribution)|this (work|study|paper)|our |the results?|"
                          r"it (is|was) (found|shown)|these results)", re.I)


def first_sentences(text: str, n: int) -> str:
    """Motivation sentences only: stop before the first sentence that reports what the authors did or found."""
    text = re.sub(r"^\s*abstract\s*", "", (text or "").strip(), flags=re.I)
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if RESULT_START.match(sentence) or len(kept) == n:
            break
        kept.append(sentence)
    return " ".join(kept).strip() or "(not available)"


def build_item(record: dict[str, Any], flaws: list[Flaw], item_id: str) -> dict[str, Any]:
    text_all = f"{record.get('title', '')} {record.get('abstract', '')}"
    ctx = infer_context(text_all)
    planted_dims = {f.dimension for f in flaws}
    approach = []
    for (dim, c), sentence in SOUND.items():
        if c != ctx:
            continue
        approach.append(next(f.flawed for f in flaws if f.dimension == dim) if dim in planted_dims else sentence)
    for f in flaws:  # flaws whose dimension has no sound sentence in this context
        if f.flawed not in approach:
            approach.append(f.flawed)
    text = (f"Aim: {record['title'].rstrip('.')}.\n\n"
            f"Background: {first_sentences(record.get('abstract', ''), 2)}\n\n"
            f"Approach: {' '.join(approach)}")
    return {"id": item_id, "kind": "planted", "context": ctx, "source_doi": record.get("doi"),
            "text": text, "planted": [{"flaw_id": f.id, "dimension": f.dimension, "description": f.description}
                                      for f in flaws]}


def make_items(records: list[dict[str, Any]], n: int, max_flaws: int, clean_frac: float, seed: int
               ) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    usable = [r for r in records if r.get("role", "primary") == "primary" and r.get("title")
              and len(r.get("abstract") or "") > 300 and r.get("on_topic", True)
              and first_sentences(r.get("abstract", ""), 2) != "(not available)"]
    rng.shuffle(usable)
    items = []
    for i, rec in enumerate(usable[:n]):
        ctx = infer_context(f"{rec['title']} {rec['abstract']}")
        candidates = [f for f in FLAWS if ctx in f.contexts]
        k = 0 if rng.random() < clean_frac else rng.randint(1, max_flaws)
        chosen: list[Flaw] = []
        for f in rng.sample(candidates, len(candidates)):
            if len(chosen) < k and f.dimension not in {c.dimension for c in chosen}:
                chosen.append(f)
        items.append(build_item(rec, chosen, f"pf-{i:04d}"))
    return items


def detect(flaw: Flaw, critique: str) -> bool:
    """Cheap first-pass check; the LLM judge in critique_eval.py is the real scorer."""
    return all(re.search(group, critique, re.I) for group in flaw.detect)


def selftest() -> None:
    rec = {"title": "Ultrathin MFI nanosheet membranes for xylene separation", "doi": "10.0000/x.1",
           "abstract": "Zeolite membranes could cut separation energy. Thin selective layers are needed. "
                       "Here we report something. " * 10}
    item = build_item(rec, [FLAW_BY_ID["mismatched_benchmark"], FLAW_BY_ID["single_batch"]], "t1")
    assert item["context"] == "membrane"
    assert "regardless of the temperature" in item["text"] and "one synthesis batch" in item["text"]
    assert "at matched temperature" not in item["text"]  # sound sentence replaced
    good = "Weakness: benchmarking against literature values measured at different feed pressure is unfair."
    assert detect(FLAW_BY_ID["mismatched_benchmark"], good)
    assert not detect(FLAW_BY_ID["mismatched_benchmark"], "The membrane looks promising.")
    items = make_items([dict(rec, doi=f"10.0000/x.{i}") for i in range(30)], n=20, max_flaws=2,
                       clean_frac=0.25, seed=0)
    assert len(items) == 20 and any(not it["planted"] for it in items) and any(len(it["planted"]) == 2 for it in items)
    for (dim, _), _s in SOUND.items():
        assert dim in {f.dimension for f in FLAWS}, dim
    print("selftest passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("make")
    s.add_argument("--works", type=Path, required=True, help="works.jsonl from build_expert_corpus.py")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--max-flaws", type=int, default=2)
    s.add_argument("--clean-frac", type=float, default=0.2, help="share of items with no planted flaw")
    s.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest")
    args = p.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "list":
        for f in FLAWS:
            print(f"{f.id:<26} {f.dimension:<28} {','.join(f.contexts):<28} {f.description}")
    else:
        records = [json.loads(l) for l in args.works.open(encoding="utf-8") if l.strip()]
        items = make_items(records, args.n, args.max_flaws, args.clean_frac, args.seed)
        with args.out.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"wrote {len(items)} items ({sum(not i['planted'] for i in items)} clean controls) to {args.out}")


if __name__ == "__main__":
    main()
