# Building a Catalysis Research Agent with Agentic RL: Requirements, Data and Development Plan

This plan is for a chemical-engineering researcher working on catalysis who wants to build a research agent and improve it with agentic RL. It builds on this repo: the multi-turn skeleton from `experiments/search-r1`, the code sandbox from `experiments/retool`, LLM-judge rewards from `experiments/tournament-grpo`, the SFT→RL pipeline from `experiments/spec-o3`, and the Tinker port in [`../tinker/`](../tinker/README.md).

> **Zeolite synthesis edition:** see [`zeolite-plan.md`](./zeolite-plan.md) for the version tailored to zeolite synthesis: proposal copilot, lab-data tools, data counts and local deployment.

---

## 1. Summary

1. **RL is the last step, not the first.** Most of a research agent's quality comes from its tools (literature search, databases, a Python sandbox with chemistry libraries), its corpus and its evaluation set. Build those first and measure a prompted baseline. RL then teaches a smaller model to *use* them well: when to search, which tool to call, how to check units, when to stop. It also makes that model cheap enough to run thousands of times.
2. **Train on tasks whose answers can be checked.** RL only works where you can compute a reward. In catalysis that means numeric facts from papers, engineering calculations, database lookups and structured extraction. Open-ended "write me a review" tasks can use judge rewards later, but they are much easier to game (lab 05).
3. **The most valuable thing you can make is an expert-curated evaluation set** of 150–300 questions. It defines success, catches reward hacking and is hard to outsource. Most *training* data can be generated semi-automatically.
4. **Start narrow.** Pick one sub-field (for example CO₂ hydrogenation to methanol, NH₃ synthesis, or OER electrocatalysts) and broaden only after the full pipeline works there.

---

## 2. What the agent should do

| # | Capability | Example | Tools | Checkable reward? | RL priority |
|---|---|---|---|---|---|
| C1 | **Literature fact QA** | "Highest methanol selectivity reported for In₂O₃/ZrO₂ at 300 °C, 5 MPa?" | `search_literature`, `read_paper` | ✅ match within numeric tolerance + cited DOI | **High** (≈ Search-R1) |
| C2 | **Engineering calculations** | TOF from rate, loading and dispersion; Eₐ from an Arrhenius fit; equilibrium conversion; Weisz–Prater / Mears criteria; GHSV ↔ WHSV | `python` (numpy, scipy, Cantera, pint) | ✅ compare to the reference solution | **High** (≈ ReTool) |
| C3 | **Database lookup and reasoning** | "Which of these Cu-based alloys has a CO binding energy closest to the volcano optimum?" | `query_catalysis_hub`, `query_materials_project`, `python` | ✅ ground truth from the database | Medium–High |
| C4 | **Structured extraction** | Turn a paper into a record: catalyst, support, synthesis, T, P, feed, GHSV, conversion, selectivity, stability | `read_paper` | ✅ field-level F1 against expert annotations | Medium (also feeds C1/C3) |
| C5 | **Comparison and screening** | "Rank 5 candidate promoters for Ni/Al₂O₃ dry reforming by coke resistance, with evidence" | all of the above | ⚠️ partly (ranking vs. expert/DB) | Medium |
| C6 | **Hypothesis / experiment proposal** | "Propose the next 3 experiments to raise C₂₊ selectivity" | all + Bayesian optimization (BO) helper | ❌ judge/rubric only | Later |
| C7 | **Research report** | A two-page evidence-backed summary with citations | all | ❌ pairwise judge (Tournament-GRPO) | Later |

Recommended order: **C2 + C1 first**, then C4 (which also builds your database), then C3/C5, then C6/C7.

---

## 3. Architecture

```
                         ┌──────────────────────── Environment (your code) ────────────────────────┐
 question ─► policy LLM ─┤ search_literature(query)  → hybrid BM25 + embedding index over your corpus │
 (Qwen 4–8B on Tinker,   │ read_paper(doi, section)  → parsed full text / tables                      │
  later SFT+RL'd)        │ python(code)              → sandbox: numpy, scipy, pandas, pint, Cantera,  │
        ▲                │                             pymatgen, ASE, RDKit (from retool/sandbox.py)   │
        │                │ query_db(...)             → Catalysis-Hub, Materials Project, your lab DB  │
        │                │ (later) predict_ads(...)  → ML interatomic potential (Open Catalyst models)│
        │                └─────────────────────────────────────┬────────────────────────────────────┘
        │                                                      ▼
        │                     final "Answer: … [doi:…]"  ─►  reward.py (tolerance / F1 / citation / judge)
        └────────────── GRPO / OPD update (train.py, Tinker) ◄──┘
```

Design rules (taken from the study guide):
- **Use a local, cached retrieval index for RL.** Training makes thousands of tool calls. Live web search is slow, costly and non-deterministic, so keep it for deployment only.
- Keep tool outputs **short and truncated** (`fit_tool_content` in `search-r1/rollout.py`), and mask them out of the loss (`search-r1/train.py:62`).
- Every tool returns **sources** (a DOI, or a database ID plus URL) so answers can be checked.
- Use one `Answer:` line for machine-checkable tasks and add a citation field: `Answer: 78 % [doi:10.xxxx/yyyy]`.

---

## 4. What you need to prepare

### 4.1 Skills (beyond the study guide)
| Area | What | Why |
|---|---|---|
| Python data stack | pandas, numpy, scipy, **pint** (units) | calculation tools and reward checking |
| Chemistry informatics | **pymatgen**, **ASE**, **Cantera** (thermo/kinetics), RDKit (if molecular) | sandbox tools, task generation |
| Information retrieval | chunking, BM25, embeddings, reranking (e.g. `rank_bm25`, sentence-transformers, FAISS/Chroma) | the `search_literature` tool |
| PDF parsing | GROBID, Marker or Nougat for text; separate table extraction | building the corpus |
| Catalysis databases | Catalysis-Hub (GraphQL API), Materials Project (`mp-api`), Open Catalyst (OC20/OC22) | C3 tools and tasks |
| Evaluation | building held-out sets, inter-annotator agreement, error analysis | trustworthy results |

### 4.2 Accounts and infrastructure
| Item | Notes |
|---|---|
| **Tinker** credits | policy sampling and training (you have this) |
| A strong LLM API (optional) | prompted baseline, teacher trajectories, judging. **Check the provider's terms on using outputs to train models.** An open-weight teacher hosted on Tinker avoids that issue and is required for OPD anyway (same tokenizer). |
| Literature APIs | OpenAlex (free), Semantic Scholar API (free key), Crossref; arXiv/ChemRxiv; PubMed Central OA subset; publisher text-and-data-mining (TDM) APIs (Elsevier, Springer Nature, Wiley) through your **institutional licence** |
| Materials Project API key | free |
| Workstation | 16–32 GB RAM and 50–200 GB disk for the corpus and indexes. A GPU is optional: it helps with embedding a large corpus and with ML potentials, and isn't needed for RL (that runs on Tinker). |
| Tracking and versioning | git + DVC (or similar) for datasets; SwanLab/W&B for runs |

### 4.3 People
- **You, as domain expert:** about 2–4 h/week writing and checking eval questions and reviewing failure cases.
- **Ideally a second annotator** (labmate or student) for about 20 % overlap on the eval set, so you can measure agreement. If two experts disagree on an answer, the model can't be graded on it.

### 4.4 Legal and ethics
- Paper full texts are copyrighted. Index only open-access papers or papers your institution's TDM licence covers, and keep the corpus private.
- Don't put unpublished lab data into third-party APIs without permission.
- Add safety notes for hazardous procedures (high-pressure H₂, pyrophoric catalysts, CO).

---

## 5. Data: what and how much

The amounts below assume one narrow sub-field and the repo's scale of experiments (Qwen 4–8B, LoRA, 20–200 RL steps × 8–32 prompts per step × group size 8). "Minimum" is enough to learn and get a signal. "Recommended" is enough for a result you could publish.

| # | Dataset | Purpose | Minimum | Recommended | How to build |
|---|---|---|---|---|---|
| D1 | **Literature corpus** | what `search_literature` searches | 1,000 papers (abstracts + OA full texts) | 5,000–20,000 papers | OpenAlex/Semantic Scholar queries for your sub-field → download OA PDFs → GROBID/Marker → chunk 300–800 tokens with DOI and section metadata |
| D2 | **Expert evaluation set** (gold) | the scoreboard; never trained on | 100 questions | **150–300**, across C1–C4 | You write or verify each one. Split dev (40 %) / test (60 %). Prefer papers **published after the base model's training cutoff** to limit memorization. |
| D3 | **RL training prompts** (verifiable) | GRPO prompts | 1,000 | 3,000–10,000 | Mostly semi-automatic (see 5.2). Each prompt is used 1–3 times. Filter by difficulty (5.3). |
| D4 | **SFT warm-start trajectories** | teach the tool-call format and basic habits before RL | 300 | 1,000–3,000 | Teacher model runs the agent on D3 prompts; **keep only trajectories with correct answers** (rejection sampling); spot-check 5–10 % |
| D5 | **Extraction gold annotations** | C4 reward + eval | 50 papers | 150–300 papers | You annotate with a fixed schema (5.4). This seeds D6. |
| D6 | **Structured catalyst database** | `query_db` tool + C1/C3 question source | from D5 + LLM extraction of 500 papers | 2,000–10,000 records | Run extraction over D1, validate (units, ranges), keep provenance (DOI, table/figure) |
| D7 | **External DFT/materials data** | C3 tasks and tools | a Catalysis-Hub subset for your reaction | + Materials Project, OC20/OC22 subsets | API downloads; cache locally |
| D8 | **Your own lab data** (optional, highest value) | real-world grounding, active-learning loop | — | whatever exists | Export from notebooks/ELN into the D6 schema |

**Rough proportions for D3 (1,000–10,000 prompts):** about 40 % C2 calculations (generated, unlimited), 35 % C1 literature QA, 15 % C3 database tasks, 10 % C4 extraction.

### 5.1 Task record format (JSONL)

```json
{"id": "c1-000123", "type": "lit_qa_numeric",
 "question": "What CO2 conversion was reported for 5 wt% Cu/ZnO/Al2O3 at 250 °C, 3 MPa, H2/CO2 = 3, GHSV 6000 mL g-1 h-1 in doi:10.xxxx/yyyy?",
 "answer": {"value": 12.4, "unit": "percent", "rel_tol": 0.05},
 "evidence": [{"doi": "10.xxxx/yyyy", "loc": "Table 2"}],
 "split": "train", "difficulty": null, "source": "auto+expert_checked"}

{"id": "c2-004511", "type": "calc",
 "question": "A 1.0 wt% Pt/Al2O3 catalyst (dispersion 45 %) converts CO at 3.2e-6 mol s-1 per gram of catalyst. Compute the TOF in s-1.",
 "answer": {"value": 0.1387, "unit": "1/s", "rel_tol": 0.02},
 "solution_code": "M_Pt=195.08; n_surf=0.01/M_Pt*0.45; print(3.2e-6/n_surf)",
 "split": "train", "source": "generator:tof_v1"}

{"id": "c4-000042", "type": "extraction", "doi": "10.xxxx/zzzz",
 "schema": "catalyst_record_v1",
 "gold": [{"catalyst": "In2O3/ZrO2", "loading_wt_pct": 9, "T_C": 300, "P_MPa": 5,
           "H2_CO2": 4, "GHSV": 16000, "GHSV_unit": "h-1",
           "X_CO2_pct": 5.2, "S_MeOH_pct": 99.8, "TOS_h": 1000}]}
```

### 5.2 Generating training data at scale (so you don't write 5,000 questions by hand)

1. **C2 calculations: write generators.** Each generator is a Python function that samples realistic parameters, writes the question and computes the answer. Start with:
   - TOF from rate, loading and dispersion
   - Arrhenius / Eₐ from rate–temperature data (with noise)
   - Equilibrium conversion (Cantera or ΔG data)
   - GHSV/WHSV/space-time conversions
   - Weisz–Prater and Mears criteria
   - Thiele modulus and effectiveness factor
   - Langmuir–Hinshelwood rate fitting
   - Selectivity, yield and carbon balance
   - Degree of rate control from a microkinetic model (advanced)

   Each generator yields effectively unlimited, perfectly labelled tasks. It's the cheapest data you will ever get.
2. **C1 literature QA: generate, then verify.** An LLM reads one chunk or table and proposes (question, answer, unit, evidence location). Automatic filters keep a pair only if the answer is a number or entity in the chunk, the question names enough conditions to be unique, and an independent LLM answering *with* the chunk agrees. Then you **spot-check 10–20 %**. If more than 10 % of the spot-check is wrong, fix the pipeline, not the individual items.
3. **C3 database tasks: build from tables.** Queries over Catalysis-Hub/OC data, e.g. "Which surface among {…} has ΔE(CO*) closest to −0.67 eV?" The ground truth comes from the database. (−0.67 eV is only a placeholder; pick descriptor targets from your sub-field's literature.)
4. **Paraphrase** about 20 % of questions so the model learns the skill rather than the wording.

### 5.3 Difficulty filtering (lessons from lab 01 / 03)
Run the **base policy** 8 times on every candidate prompt (cheap compared with training):
- pass rate = 0/8 → too hard for now. Hold it back for after SFT, or check whether the label is wrong.
- pass rate = 8/8 → too easy, no learning signal (degenerate group).
- **Keep prompts with 1/8 to 7/8** for RL. This one filter often matters more than any hyperparameter.

### 5.4 Data hygiene
- **Split by paper (DOI) and by year**, never by question. Questions from the same paper must not appear in both train and test.
- Keep a units vocabulary (`pint`) and a synonym table (e.g. "Cu/ZnO/Al₂O₃" = "CZA"; "GHSV" vs "WHSV" must stay distinct).
- Record provenance for every label (DOI, table/figure, who verified it).
- Version datasets. Never edit the test set after looking at model outputs.

---

## 6. Reward design

| Task | Reward | Notes |
|---|---|---|
| C1 numeric | `1` if within `rel_tol` after unit conversion, else `0`; `−0.1` bad format | parse with `pint`. Don't give partial credit for "close" values at first (lab 03 `--partial-credit` showed the proxy-gaming problem). |
| C1 entity | normalized exact match against an alias list | synonym table from 5.4 |
| Citation bonus | `+0.2` if a cited DOI is in `evidence`; `−0.2` if a cited DOI **is not in the corpus** (hallucinated) | gives grounding without dominating the main reward |
| C2 calc | `1` if within `rel_tol` of the reference | tolerance 1–2 %, units checked |
| C3 DB | exact match / ranking metric (top-1 or Kendall τ) | |
| C4 extraction | field-level F1, with numeric fields counted correct within tolerance | punish invented fields (precision) as much as missed ones (recall) |
| C6/C7 open-ended | pairwise LLM-judge tournament with an **expert-written rubric** (`tournament-grpo/reward.py`) | only after C1–C4 work; always report the strict metrics alongside |
| Efficiency | small per-tool-call cost **switched on later** | lab 04: a cost from step 0 stops tool use from emerging |

Before training, **try to hack your own reward**: answer with every number in the chunk, cite every DOI, output units only. If any trick scores above zero, fix the reward (lab 05).

---

## 7. Development plan (about 20 weeks part-time; compresses to about 10 full-time)

Each phase has deliverables and an **exit criterion**. Don't move on until it's met.

### Phase 0 (weeks 1–2): Scope and foundations
- Do study-guide Phases 0–2 (labs 01–03; GRPO on Tinker via `study-guide/tinker/grpo_tinker.py`).
- Choose **one sub-field** and write a one-page scope: which reactions, catalyst families, condition ranges, and which capabilities (start with C1 + C2).
- Write **30 seed questions** by hand: what you'd actually ask an assistant in a week of research.
- Define metrics: C1 accuracy, C2 accuracy, citation precision, hallucinated-DOI rate, average tool calls, cost per question.
- **Exit:** scope doc + 30 seed questions with answers and evidence.

### Phase 1 (weeks 2–5): Corpus and retrieval (D1, D7)
- Harvest metadata (OpenAlex/Semantic Scholar) → download OA/TDM PDFs → parse (GROBID/Marker) → chunk with metadata.
- Build a hybrid index (BM25 + embeddings, with optional reranking). Make sure tables are searchable; numbers often live there.
- Download and cache the Catalysis-Hub / Materials Project subsets for your reactions.
- **Retrieval eval:** for the 30 seed questions, is the gold evidence chunk in the top 5 results (recall@5)?
- **Exit:** recall@5 ≥ 70 % on seed questions. If it's lower, fix chunking, tables or the query expansion first. RL can't learn to find evidence the index can't return.

### Phase 2 (weeks 4–7): Tools, environment and a prompted baseline
- Implement the tools: `search_literature`, `read_paper`, `python` (extend `retool/sandbox.py` with pint, Cantera, pymatgen, ASE), `query_db`.
- Generalize `search-r1/protocol.py` from one tool to several (tool schema list, parser, per-tool budgets). Reuse `search-r1/rollout.py` (concurrency, truncation) and `search-r1/train.py:build_datum` (mask) unchanged.
- Write `reward.py` (section 6) with unit tests, including your own hacking attempts.
- Build the eval harness: run any model through the same environment on D2.
- Run a **prompted baseline**: (a) a strong model and (b) your chosen small Tinker model, both with tools.
- **Exit:** working end-to-end loop; baseline numbers for both models; the error analysis tagged by cause (retrieval miss / wrong tool / unit error / reasoning / hallucination).

### Phase 3 (weeks 5–9): Datasets (D2–D6)
- Grow the eval set to **150+** (D2): seed questions + verified generated questions + some from your own expertise. Have a second annotator double-check about 20 %.
- Write **5–8 C2 generators** and produce about 2,000 calculation tasks.
- Run the C1 generate-and-verify pipeline → about 1,500 QA tasks; spot-check.
- Annotate **50 papers** for extraction (D5); run extraction over 500 papers → D6.
- Difficulty-filter everything with the base policy (5.3).
- **Exit:** frozen D2 test set; ≥ 1,000 filtered RL prompts; spot-check error rate below 10 %.

### Phase 4 (weeks 8–10): SFT warm start (D4), following `experiments/spec-o3/train_sft.py`
- Teacher model generates trajectories on training prompts; keep the correct ones; aim for 500–2,000.
- LoRA SFT the small model on Tinker (`cross_entropy` loss, assistant tokens only, same mask as RL).
- **Exit:** the SFT model beats the prompted small model on D2-dev; tool-call format errors below 2 %.

### Phase 5 (weeks 10–15): Agentic RL
- GRPO with the multi-tool environment, starting from the SFT checkpoint. Start with the repo's Search-R1 settings: group size 8, 8–16 prompts/step, 20-step pilot, then 100–200 steps.
- Watch: train reward, **D2-dev accuracy**, `frac_degenerate`, tool calls per question, hallucinated-DOI rate, response length.
- Ablations (one at a time): C2-only vs C1+C2 mix; citation bonus on/off; tool-call cost on later; group size 8 vs 16; PPO vs CISPO.
- Re-run the difficulty filter every ~50 steps (prompts the model has mastered stop giving signal).
- **Exit:** a statistically clear gain on D2-dev over the SFT model (report mean ± CI over 3 eval samples); no rise in hallucinated DOIs.

### Phase 6 (weeks 13–17): Long-horizon and quality improvements (optional)
- **OPD** from a larger Tinker model (same tokenizer) on the agent's own trajectories, for denser signal (`general-opd`).
- **Turn-level credit** for long research sessions (AgentOPSD ideas) once trajectories exceed about 10 tool calls.
- Start C5/C6/C7 with rubric-based pairwise judging (Tournament-GRPO); validate the judge against your own preferences on 50 pairs before trusting it.

### Phase 7 (weeks 16–20): Final evaluation, deployment and the research loop
- Evaluate **once** on the D2 test set. Compare: strong prompted model, small prompted, small+SFT, small+SFT+RL. Report cost per question too.
- Error analysis on 50 failures; write it up (the repo's chapters are a good template).
- Deploy: a CLI, notebook helper or MCP server that calls your tool layer + the trained policy (Tinker sampling, or export the LoRA and serve locally with vLLM).
- Research loop: log real questions you ask the agent → expert-verify answers → add them to the eval or training pools. Connect lab results (D8) and BO for experiment proposals (C6).

### Milestone summary

| Week | Milestone |
|---|---|
| 2 | scope + 30 seed questions; GRPO runs on Tinker |
| 5 | corpus indexed, retrieval recall@5 ≥ 70 % |
| 7 | end-to-end agent + baseline report |
| 9 | frozen eval set (150+), ≥ 1,000 RL prompts |
| 10 | SFT model beats prompted small model |
| 15 | RL model beats SFT model on dev |
| 20 | final test-set report, deployed assistant |

---

## 8. Budget: estimating Tinker tokens before you spend

Per RL step, roughly:

```
sampled_tokens ≈ prompts_per_step × group_size × avg_generated_tokens_per_trajectory
prefill_tokens ≈ prompts_per_step × group_size × avg_total_trajectory_tokens   (prompt + tool outputs, re-read each turn)
train_tokens   ≈ prompts_per_step × group_size × avg_total_trajectory_tokens
```

Example (Search-R1-like): 8 prompts × 8 samples × about 6k total tokens ≈ **0.4M tokens trained per step**, so a 100-step run is about 40M trained tokens plus similar prefill and eval. Multiply by your Tinker price list. Do a 2-step smoke run and read the actual usage before scaling. Biggest savers: difficulty filtering (no wasted degenerate groups), shorter tool outputs, a 20-step pilot before every long run.

---

## 9. Evaluation checklist

- Headline metrics on the frozen D2 test set: accuracy for C1/C2/C3, F1 for C4, **citation precision**, **hallucinated-DOI rate**, unit-error rate, tool calls and cost per question.
- 3 samples per question at temperature > 0 → report the mean and a confidence interval.
- A "post-cutoff" slice (papers newer than the base model) to separate retrieval skill from memorization.
- Human review of 30 random answers per model for things the metrics miss: misleading phrasing, cherry-picked evidence.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hallucinated numbers or citations | citation reward + hallucinated-DOI penalty; answers must quote the evidence location |
| Unit and condition confusion (GHSV vs WHSV, wt% vs at%) | `pint` everywhere; generators that deliberately mix units; condition fields in questions |
| Ambiguous questions (several papers report different values) | each question must specify the paper or full conditions; second-annotator agreement check |
| Train/test leakage | split by DOI and year; dedupe near-duplicate questions |
| Reward hacking | hack-your-own-reward tests; strict held-out metric logged every eval |
| Retrieval is the bottleneck | Phase 1 exit criterion; error analysis tags retrieval misses separately |
| Copyright / licences | OA + institutional TDM only; private corpus |
| Cost overruns | token formula (section 8), pilot runs, difficulty filtering |
| Scope creep | one sub-field until Phase 7 |

---

## 11. Decisions to make now

1. **Sub-field:** which reaction(s) and catalyst families? Thermocatalysis, electrocatalysis or photocatalysis?
2. **Computational or experimental focus?** If computational (DFT/microkinetics), C3 and ML-potential tools matter more. If experimental, focus on C1/C4 and lab data.
3. **Which capabilities first?** Recommended: C2 + C1.
4. **Literature access:** which publishers does your institution's TDM licence cover?
5. **Teacher model:** a strong API model for SFT data (check its terms) or a large open-weight model on Tinker (also usable for OPD)?
6. **Policy model:** run `grpo_tinker.py --list-models` and pick a 4–8B instruct model for pilots.

---

## 12. Where each piece comes from in this repo

| Component | Start from |
|---|---|
| Multi-turn rollout, concurrency, truncation | `experiments/search-r1/rollout.py` |
| Tool protocol, token-exact prompt extension | `experiments/search-r1/protocol.py` |
| Loss mask / datum building | `experiments/search-r1/train.py:62` |
| Code sandbox | `experiments/retool/sandbox.py` |
| Math-equivalence style answer checking | `experiments/retool/reward.py` |
| SFT warm start → RL | `experiments/spec-o3/train_sft.py`, `train_rl.py` |
| LLM-judge tournament rewards | `experiments/tournament-grpo/` |
| On-policy distillation | `experiments/general-opd/` |
| Turn-level credit | `experiments/agentopsd/advantages.py` |
| Tinker backend | `study-guide/tinker/` |
