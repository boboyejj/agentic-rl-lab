# Distilling Expert Knowledge into the Zeolite Research Agent

This extends the [zeolite plan](./zeolite-plan.md) (Tracks A, B, C) and the [general catalysis plan](./README.md). It makes "learning from leading heterogeneous-catalysis researchers" concrete: who to include, how to collect and process their work, how to turn it into rubrics and training examples, how much data you need, and how to spend your advisor's time well.

Code for this file is in [`expert-distillation/`](./expert-distillation/):

| File | What it does |
|---|---|
| `build_expert_corpus.py` | author names → OpenAlex profiles → all works, classified as *primary* / *opinion* (review, perspective, account) / *thesis*, with abstracts, OA links and licences |
| `rubric_tools.py` | opinion docs → rubric items and reviewer questions (each with a verified quote and DOI) → Tournament-GRPO judge format; blinded A/B sheets for your advisor; judge–expert agreement |
| `experts.example.toml` | zeolite expert candidates (verify before use) |
| `experts.saf.example.toml` | SAF expert candidates for the later extension ([saf-plan.md](./saf-plan.md)) |

Both scripts use only the Python standard library and have an offline `selftest`.

---

## 1. What "distilling an expert" means here

An expert's value has four layers. Only the first three can come from published work.

| Layer | Example (zeolites) | Where it comes from | How it enters the agent |
|---|---|---|---|
| **Knowledge** | which gel compositions give MFI nanosheets; reported separation factors | research articles, theses | cited retrieval corpus (Track A + B), factual QA tasks with verifiable rewards |
| **Criteria** | "thickness must be measured directly, not inferred from XRD broadening" | reviews, perspectives, accounts, outlook sections | **rubric bank** → judge for critiques and reports |
| **Habits** | the questions they always ask; how they structure a critique | perspectives, theses' "future work", (with consent) talks and interviews | **SFT examples** of evidence-tied critiques; reviewer-question tasks |
| **Judgement** | which of two plausible aims is more promising | only from the person | **real feedback**: preference judgements, annotated critiques, eval answers |

Distillation in the RL sense (OPD/OPSD) is one *training method* you may use later (section 8.3). It copies a teacher *model*, not a person.

### Ground rules (keep these in the repo and in the system prompt)
1. **No impersonation.** The agent never speaks as "Prof. X" and never says "X would think…". It says "Reviews by X and co-workers (doi:…) emphasize…".
2. **Every criterion carries a source**: DOI plus a verbatim quote (`rubric_tools.py merge` drops items whose quote is not in the source).
3. **Published or consented material only.** Licensed full text only under your library's TDM terms; nothing unpublished (talk recordings, reviews they wrote for you, emails) without explicit consent.
4. **Style is not judgement.** A model trained on expert-style critiques sounds expert. Only a held-out evaluation set graded by real experts (section 6) tells you whether it is right.
5. **Keep the corpus private** (see `.gitignore` in `expert-distillation/`).

---

## 2. Choosing experts and groups

### 2.1 Score candidates, don't just pick famous names
Score each candidate 0–3 on five criteria and keep the 10–20 highest. Fill this in as a spreadsheet after running `resolve` and `harvest` (the `summary.csv` output gives most numbers).

| Criterion | 0 | 3 | How to measure |
|---|---|---|---|
| **Theme fit** | other field | publishes on ≥ 2 of your themes (2D zeolites, MFI nanosheets, OSDA-free/direct syntheses, membranes, xylene separation) | `on_topic` share in `summary.csv`; topic list from `resolve` |
| **Opinion density** | no reviews | ≥ 5 reviews/perspectives/accounts since 2010 | `opinion` column; this sets your **rubric yield** |
| **Distinct viewpoint** | duplicates someone already chosen | covers a dimension nobody else does (e.g. crystallization mechanism, membrane scale-up, computation, industry) | your judgement; aim for coverage of all 8 rubric dimensions (section 4.1) |
| **Recency** | inactive since 2015 | active in the last 3 years | year distribution in `works.jsonl` |
| **Access** | mostly closed, not in your TDM licence | high OA share or licensed publisher | `open_access` column + library answer |

### 2.2 Three tiers
| Tier | Who | Count | Weight | Can give feedback? |
|---|---|---|---|---|
| **1. Your group** | advisor, senior labmates, group theses | 1–3 | highest | yes, with consent |
| **2. Core field** | groups publishing directly on your themes | 6–10 | normal | no (published work only) |
| **3. Adjacent** | characterization, kinetics, computation, scale-up, industry | 4–8 | for specific rubric dimensions | no |

`experts.example.toml` lists starting candidates for tiers 2 and 3 (e.g. groups known for 2D zeolites and membranes, crystallization mechanisms, hierarchical zeolites, OSDA design and active-site characterization). Treat it as a draft: **verify every profile** and adjust to your advisor's view of the field.

### 2.3 Find the ones you would miss
- **Who cites your group, and whom your group cites:** OpenAlex `cites:` / `cited_by` filters on your group's 20 most important papers; count authors.
- **Co-authorship graph:** frequent co-authors of tier-2 experts.
- **Special-issue editors and review authors** in *Microporous and Mesoporous Materials*, *Chem. Rev.*, *Chem. Soc. Rev.*, *Acc. Chem. Res.* on zeolites.
- **Industry:** patents (PatentsView) on nanosheet/membrane routes; industrial authors bring scale-up and cost criteria academics underweight.
- **Diversity check:** don't take 10 people from one school of thought; the rubric bank should contain disagreements, labelled as such.

---

## 3. Corpus pipeline

```
experts.toml ─► resolve ─► (you confirm IDs) ─► harvest ─► works.jsonl ─┬─► opinion_docs.jsonl ─► rubric_tools prompts/run/merge ─► rubric_bank.jsonl
  (names)       OpenAlex                          OpenAlex   (+ enrich:   │                                                                  │
                author search                     /works     S2 TLDRs)   ├─► fetch_queue.csv ─► legal PDF download ─► GROBID/Marker ─► chunks ─► search index (Track A/B)
                                                                         └─► primary works ─► factual QA + extraction tasks (Track B, verifiable rewards)
```

### 3.1 Commands
```bash
cd study-guide/catalyst-research-agent/expert-distillation
cp experts.example.toml experts.toml          # private; edit tier-1 entries
export OPENALEX_MAILTO="lab-contact@your-university.edu"   # polite pool; use a project address
python build_expert_corpus.py resolve --config experts.toml --out corpus/
#   -> read the candidate list; paste every correct ID into openalex_ids (OpenAlex splits some people)
python build_expert_corpus.py harvest --config experts.toml --out corpus/
python build_expert_corpus.py enrich --out corpus/        # optional: Semantic Scholar TLDRs, review flags
```

### 3.2 Triage before anything goes into training
1. **Off-topic filter:** `on_topic=false` works (matched none of the expert's `topic_terms`) stay out of the rubric queue but remain searchable.
2. **Role check:** skim `opinion_docs.jsonl` titles (≈ 10 min for 300 docs). The title/venue heuristics catch most reviews and perspectives; fix misclassifications by hand.
3. **Author position:** `expert_positions` tells you whether the expert is first, middle or last author. For criteria attributed to an expert's *group*, prefer first/last-author reviews.
4. **Licences:** `fetch_queue.csv` lists OA PDFs with their licence. CC-BY can be stored and processed; for anything else, follow your library's TDM answer.

### 3.3 Full text and processing (same stack as Track B, D1)
- Download OA PDFs politely (one at a time, with pauses), plus licensed PDFs from your Zotero library.
- Parse with GROBID or Marker; keep section headers. For opinion docs, the most valuable sections are the **introduction, "challenges", "outlook"/"perspective" and conclusions**.
- Save plain text as `corpus/text/<sanitized-doi>.txt` so `rubric_tools.py prompts --fulltext-dir corpus/text` uses it instead of the abstract.
- Index all chunks with metadata `{doi, year, experts, role, section}`. The `experts` field lets Track A answer "what do recent reviews from groups working on membranes say about…" with citations.

### 3.4 Expected yield (orders of magnitude; your `summary.csv` gives the real numbers)
| Item | Per expert | 15 experts |
|---|---|---|
| works since 2000 | 100–500 | 2,000–5,000 unique (overlap from co-authorship) |
| on-topic works | 50–70 % | 1,000–3,000 |
| opinion docs (reviews, perspectives, accounts) | 5–30 | **150–400** |
| open access | 30–60 % | depends on publishers |

---

## 4. Rubrics: from reviews to a judge you can trust

### 4.1 Dimensions
`rubric_tools.py` uses a fixed vocabulary so items from different experts can be compared and balanced:

| Dimension | Typical expert criterion (illustrative) |
|---|---|
| `characterization_evidence` | thickness/morphology measured directly (AFM, TEM) on statistically meaningful samples |
| `synthesis_reproducibility` | full gel composition, reagent sources, aging, seeding, yield reported |
| `mechanism_claims` | formation mechanism supported by time-resolved data; alternatives discussed |
| `structure_property_link` | defects, orientation, Si/Al actually linked to performance |
| `performance_benchmarking` | separation factor/permeance compared at matched conditions; stability tested |
| `scalability_cost` | OSDA cost, batch-to-batch reproducibility, membrane area, long-term operation |
| `novelty_positioning` | clear delta over closest prior work |
| `safety_sustainability` | HF/fluoride use, OSDA waste, energy use |

Edit `DIMENSIONS` in `rubric_tools.py` if your advisor groups things differently. Do it **before** extraction.

### 4.2 Where criteria are written down
1. **Reviews, perspectives, accounts** by your experts (the main source).
2. **Outlook and "challenges" sections** of research papers (often more pointed than full reviews).
3. **Theses**: "future work" and "lessons learned" chapters of your group's theses (tier 1).
4. **Published peer-review files.** Some journals (e.g. *Nature Communications*) publish reviewer reports with the article. These are anonymous, so they are *field* criteria, not attributable to a named expert. Check the licence before use.
5. **Funder review criteria** (NSF Intellectual Merit / Broader Impacts, DOE merit-review criteria): use them as the top-level structure for proposal critiques.

### 4.3 Extraction workflow
```bash
python rubric_tools.py prompts --docs corpus/opinion_docs.jsonl --fulltext-dir corpus/text --out jobs.jsonl
python rubric_tools.py run --jobs jobs.jsonl --out raw.jsonl --model qwen3:14b   # local Ollama by default
python rubric_tools.py merge --jobs jobs.jsonl --raw raw.jsonl --out rubric_bank.jsonl
```
- Published papers can be sent to an API model without the privacy concerns of your proposals, but check the publisher licence for full texts. A local model avoids the question entirely.
- `merge` **rejects items whose quote is not in the source**, items with an unknown dimension, and vague items, then merges near-duplicates and counts how many independent sources support each criterion (`support`).
- Expect 3–10 accepted items per full-text review and 0–3 per abstract.

### 4.4 Canonicalize (you), then calibrate (advisor)
1. **You** cluster the raw bank (several hundred items) into **40–80 canonical items**, each with its supporting sources. Mark disagreements between experts explicitly ("Source A recommends X; source B argues Y").
2. **Advisor, one 60–90 min session:** go through the canonical list; for each item mark keep/edit/drop and importance high/medium/low. This is the most valuable hour of expert time in the whole project. Set `expert_reviewed: true` on kept items.
3. **Freeze** as `rubric_bank_v1.jsonl`. Later changes create v2; never edit a frozen version that an evaluation used.

### 4.5 From rubric bank to judge
- For each task (a proposal aim, a paper draft, an agent report), `rubric_tools.py select` picks 6–10 relevant items, balanced across dimensions, in exactly the format `experiments/tournament-grpo/judge.py:build_judge_context` expects (`title`, `description`, `dimension`, `importance`).
- **Validate the judge before trusting it** (section 7, "judge calibration"): 50–100 blinded pairs that your advisor (or you, for a first pass) labels; target **≥ 75 % agreement and Cohen's κ ≥ 0.5** with the expert. Also compare *rubric judge* vs *no-rubric judge*; if the rubric doesn't raise agreement, the rubric needs work.
- Known judge failure modes to test: preferring longer answers, preferring confident tone, rewarding citations without checking them. Add a citation checker (cited DOIs must exist in the corpus) outside the judge.

---

## 5. Critique and reviewer-question examples (SFT data)

### 5.1 Task inputs (what gets critiqued)
| Source | Count available | Notes |
|---|---|---|
| **Mini-proposals derived from recent papers**: the teacher rewrites a paper's introduction as an aim, *without* its results | hundreds | the paper's actual results and stated limitations become grounding for the critique; prefer papers after the base model's cutoff |
| Abstracts of recent papers on your themes | thousands | "critique this study design" |
| Your own past drafts and proposal aims | a few | **local model only**; best for eval, not training |
| Synthetic flawed variants (remove a control, drop a characterization, overclaim) | unlimited | each has a **known flaw**, so critique recall is checkable automatically |

The last row is worth emphasizing: if you inject "thickness inferred only from XRD" into a mini-proposal, a good critique must mention it. That gives a **verifiable reward** for critiques ("did it find the planted flaw, citing the right criterion?") that is much harder to game than a judge score.

### 5.2 Example format (one SFT/eval record)
```json
{"id": "crit-0042", "type": "critique",
 "input": {"aim": "Synthesize sub-5 nm MFI nanosheets without OSDA by ... and fabricate p-xylene-selective membranes.",
           "context": "derived from doi:10.xxxx/yyyy (results withheld)"},
 "rubric_ids": ["r0012", "r0031", "r0045", "r0060"],
 "planted_flaws": ["thickness claimed from XRD peak width only"],
 "target": {
   "strengths": ["..."],
   "weaknesses": [{"text": "Thickness is inferred from XRD broadening only; direct AFM/TEM statistics are needed.",
                   "rubric_id": "r0012", "evidence": ["doi:10.xxxx/aaaa"]}],
   "questions": ["How will defect density in the membrane be assessed before permeation tests?"],
   "suggested_experiment": "..."},
 "provenance": {"drafted_by": "teacher", "reviewed_by": "you", "edit": "minor"}}
```
*(Placeholder DOIs; real records cite corpus documents.)*

### 5.3 Drafting and checking
1. The teacher model gets the input **plus privileged context**: the selected rubric items and 3–5 retrieved passages from expert reviews. It writes the critique in the fixed format, citing rubric IDs and DOIs.
2. Automatic filters: every cited DOI must be in the corpus; every `rubric_id` must exist; planted flaws must be found.
3. **You review** each example in ≤ 2 minutes: accept / minor edit / reject. Track the edit rate; if more than 30 % need major edits, fix the prompt or the retrieval, not the examples.
4. Reviewer-question tasks (`questions` jobs from `rubric_tools.py`) become a second task type: "list the 5 most important questions a reviewer would ask", scored by recall against the expert-derived question set for that topic.

---

## 6. How much data

| Asset | Minimum to start | Good | Effort |
|---|---|---|---|
| Experts | 8 | 12–20 | 1 day (resolve + verify) |
| Corpus (on-topic works) | 500 | 2,000–5,000 | automated |
| Opinion docs processed for rubrics | 50 | 150–400 | automated + spot checks |
| Raw rubric items (after validation) | 200 | 500–2,000 | automated |
| **Canonical rubric items** | 30 | **40–80** | you: 1–2 days; advisor: 60–90 min |
| Reviewer-question bank | 100 | 300–600 | automated + skim |
| **Critique SFT examples** | 300 | **600–1,000** | teacher + your review, ≈ 2 min each (10–30 h total) |
| Critique RL prompts (planted-flaw + rubric-judged) | 300 | 1,000–3,000 | mostly generated |
| **Held-out critique eval** | 40 | **60–150** | inputs you choose; expert-graded |
| **Judge calibration pairs** (expert-labelled) | 50 | 100–150 | advisor: 2–3 sessions of 30–45 min |
| Preference pairs for reward modelling (only if you go that way) | — | 300–1,000 | rarely worth it at this scale; prefer rubric judge + verifiable checks |
| Advisor-answered eval questions | 30 | 50–150 | spread over months, 15 min batches |

Rule of thumb: **SFT examples teach format and habits; the eval set and judge calibration decide whether any of it is right.** If you must cut, cut SFT count, never the eval set.

---

## 7. Involving your advisor and other experts efficiently

Principles: ask for **decisions, not writing**; batch requests; blind everything; give something back (they get the tool and its outputs for their own proposals).

| Ask | Their time | When | What you get |
|---|---|---|---|
| **Expert list review**: "who's missing, who's out of scope?" | 15 min | week 1 | tier lists, missed groups |
| **Rubric calibration**: keep/edit/drop + importance on 40–80 canonical items | 60–90 min, once | week 4–6 | `rubric_bank_v1` (`expert_reviewed`) |
| **Blinded A/B judgements** on critiques/answers (`rubric_tools.py pairs`): pick better, confidence, one-line reason | 30–45 min per 30–40 pairs, 2–4 sessions | weeks 6–16 | judge calibration (κ), real preference signal |
| **Eval questions**: answer or verify 10 questions per batch | 15 min batches | monthly | gold eval items with an expert's answer |
| **Annotated critiques**: mark up 2–3 agent critiques of real aims (track changes) | 30 min, a few times | after SFT | the richest feedback; also eval |
| **Think-aloud interview** (recorded and transcribed only with consent): critique one proposal aloud | 45–60 min, 1–2 times | weeks 3–8 | criteria nobody writes down; question styles |
| **Final check**: 20 random agent outputs, "would you trust this?" | 30 min | week 20 | reality check for the write-up |

Total ≈ **6–10 hours of advisor time over 20 weeks**. A labmate as second annotator on 20 % of items gives you inter-annotator agreement, the ceiling for any judge.

Other experts (tier 2/3): only if they volunteer. A short, specific ask works best, e.g. "would you rank 20 anonymized critiques of membrane proposals (30 min)?" Record consent in `experts.toml` (`consent = "participating"`) and keep their labels separate.

---

## 8. How this fits Tracks A and B, and what code to reuse

### 8.1 Track A (proposal assistant, no RL)
| Feature | Uses |
|---|---|
| novelty check with expert context | expert corpus indexed with metadata; retrieval filtered by `role=opinion` for "what do reviews say is open?" |
| **rubric-guided critique** | `select` → 6–10 rubric items → in the critique prompt; output must cite rubric IDs and DOIs |
| reviewer-question checklist | question bank filtered by topic |
| choosing between prompt variants | tournament judge + your blinded labels as an offline A/B test (no training) |

Privacy: the rubric bank and corpus are built from published work, so they can go to an API model. **Your proposal text** goes only to a model you are allowed to use (local policy for sensitive parts).

### 8.2 Track B (trained agent): three new task types
| Task | Reward | Data source |
|---|---|---|
| **B-qa (expert corpus)**: factual synthesis/membrane questions | verifiable (numeric tolerance / entity match + citation check) | primary works (section 3) |
| **B-critique**: critique an aim or study design | planted-flaw recall (verifiable) **+** rubric tournament (judge) **+** citation check | sections 4–5 |
| **B-questions**: reviewer questions for a topic | recall against question bank (judge for paraphrase matching) | `questions` jobs |

Start RL with B-qa (fully verifiable). Add B-critique only after the judge passes calibration (section 4.5).

### 8.3 Repo code to reuse
| Need | Start from | Change |
|---|---|---|
| Rubric judge, group scoring | `experiments/tournament-grpo/judge.py` (`build_judge_context`, `JudgeClient`), `reward.py` (`run_tournament`, `score_group`) | `ResearchExample.rubrics` ← `rubric_tools.py select`; point `JudgeClient` at a local or approved model; add planted-flaw recall and citation check to the score |
| Offline eval of prompted variants | `experiments/tournament-grpo/eval.py` (`generate`, `score`) | run on the held-out critique eval |
| Multi-turn rollout with search | `experiments/tournament-grpo/rollout.py`, `search.py` (replace web search with your local index) | tools: `search_corpus(query, role, experts)`, `get_rubric(topic)` |
| **SFT warm start → RL** | `experiments/spec-o3/train_sft.py`, `train_rl.py` | text-only; port to Tinker with `study-guide/tinker/README.md` |
| **Privileged-context self-distillation** | `experiments/opsd/` | **teacher = the same policy with rubric items + expert review passages in context; student = no extra context.** Same tokenizer automatically, and the "teacher knowledge" is published expert text, not a bigger model's opinions. Best fit for "expert-informed" critique. |
| Distillation from a larger model | `experiments/general-opd/`, `medical-opd/` (SFT → OPD sequence) | teacher must share the policy's tokenizer family; use for format and reasoning, not for "expert judgement" |
| Turn-level credit in long sessions | `experiments/agentopsd/` | only once sessions exceed ~10 tool calls |
| Reward hacking checks | `study-guide/labs/lab05_reward_hacking.py` | try: very long critiques, citing every DOI, generic criticisms that fit any proposal |

### 8.4 Where it lands in the 20-week timeline
| Weeks | Add to the zeolite plan |
|---|---|
| 1–2 | expert list (advisor 15 min); `resolve` + `harvest`; triage |
| 3–4 | OA download + parsing of opinion docs; rubric extraction; Track A uses raw bank |
| 5–6 | canonical rubric bank; **advisor calibration session**; freeze v1 |
| 6–9 | critique inputs + planted flaws; 300–600 SFT critiques (teacher + your review); judge calibration batch 1 |
| 9–12 | SFT on Tinker (B-qa + B-critique formats); held-out critique eval |
| 13–17 | GRPO: B-qa first, then B-critique once judge κ ≥ 0.5; optional OPSD with privileged rubric context |
| 18–20 | final expert-graded eval; write-up |

---

## 9. First steps (this week)

1. `python build_expert_corpus.py selftest` and `python rubric_tools.py selftest` (offline).
2. Copy `experts.example.toml` → `experts.toml`; add your advisor and group; remove/add candidates.
3. Run `resolve` from a network that can reach `api.openalex.org` (it was blocked from the cloud session used to write this). Confirm IDs.
4. Run `harvest`; open `summary.csv`; score candidates (section 2.1); trim to 10–15.
5. Ask your advisor the 15-minute question: "Here are 15 groups whose published work I'd use to build review criteria for the agent. Who's missing? Who would you drop?"
6. Pick 20 opinion docs you can legally use in full text; run `prompts` → `run` (local model) → `merge`; read every accepted item. This tells you how well extraction works before you scale.
7. Write 10 planted-flaw mini-proposals by hand to fix the format, before generating more.

---

## 10. Checklist before anything goes into training

- [ ] every rubric item has a DOI and verified quote; no item attributed to a person without a source
- [ ] corpus and `experts.toml` are not in any public repository
- [ ] full texts used only under OA licence or library TDM terms
- [ ] no unpublished or consented-only material in API calls without permission
- [ ] critique eval set held out **by source paper** and never used in SFT/RL
- [ ] judge calibrated against expert labels (agreement, κ) and tested against length/citation hacks
- [ ] outputs say "published work by … argues …", never "Prof. … thinks …"
