# Zeolite Synthesis Research Agent: Tailored Plan

This tailors the [general catalysis plan](./README.md) to **heterogeneous catalysis / zeolite synthesis**. The general plan still holds for data hygiene, reward design and the budget formula. This file covers what changes.

Follow-ups: [expert-distillation.md](./expert-distillation.md) (corpus, rubrics and critique data from leading groups' published work) and [saf-plan.md](./saf-plan.md) (sustainable aviation fuel extension).

Context used: Tsapatsis-group-style research themes (MFI and other framework syntheses, 2D zeolite nanosheets, OSDA-free and direct nanosheet syntheses, zeolite membranes for separations such as xylene isomers, catalytic applications).

---

## 1. Three tracks instead of one

Your needs fall into three kinds of work. Only one of them needs RL.

| Track | What it does | Needs RL? | When |
|---|---|---|---|
| **A. Proposal assistant** | novelty check against published work and patents, critique of aims, idea generation, funding search, collaborator search | **No.** A strong prompted model + tools does this well today. | **Start now** (weeks 1–4) |
| **B. Zeolite literature agent** (trained) | answer factual synthesis questions with citations: "Which OSDAs give CHA at Si/Al > 20 without fluoride?" | **Yes**, the main RL project | weeks 3–20 |
| **C. Lab-data tools** | your experiments as a searchable database + a small ML model ("which phase will this gel give?") + Bayesian optimization (BO) to propose the next experiments | **No RL.** Classic ML, called by the agent as a tool. | weeks 8–20, grows with your data |

Why A and C don't use RL:
- **Track A:** judging novelty and inventing ideas is open-ended. No reward function can reliably score "a good research idea", so RL would mostly learn to please a judge. What helps is good retrieval (papers, patents, awards) plus a capable model.
- **Track C:** fine-tuning an LLM on a few hundred experimental data points does **not** teach it chemistry trends. LLMs are poor regressors on small numeric datasets. A Gaussian process or random forest learns them far better, and the LLM agent *calls* that model as a tool (section 5).

---

## 2. Track A: proposal assistant (no training, start now)

### Tools to connect
| Need | Source | Access |
|---|---|---|
| Find related papers / check novelty | **OpenAlex** API, **Semantic Scholar** API (search + "recommended papers" + citations), Crossref | free (Semantic Scholar: free key) |
| Full text of key papers | your university library (manual download into Zotero), OA versions via Unpaywall | institutional |
| Patents (novelty matters for synthesis routes) | **USPTO PatentsView** API, Lens.org API, Google Patents (manual) | free / free tier |
| Funding opportunities | Grants.gov search API, **NSF Award Search API** (what's been funded, by which program), DOE Office of Science funding announcements, NIH RePORTER (rarely relevant) | free |
| Collaborators | OpenAlex authors/institutions + co-authorship graph, NSF awards (who holds related grants), ORCID | free |

### Workflows
1. **Novelty check:** proposal aim → 10–20 search queries (synonyms: "2D zeolite", "nanosheet", "lamellar MFI", "pillared", framework codes) → collect the top 100 papers + patents → cluster → for each aim: the closest 5 prior works, *what is actually new*, and a risk statement. **Always read the 5 closest papers yourself.**
2. **Critique:** reviewer-style review against the funder's criteria (e.g. NSF Intellectual Merit / Broader Impacts), asking for the weakest assumption in each aim.
3. **Idea generation:** cross-domain prompts ("methods that worked for MOF membranes but are untested for zeolite nanosheets"), gap mining from recent reviews, and "what failed and why" from the literature. Treat the output as brainstorming, not as conclusions.
4. **Funding match:** aim keywords → NSF Award Search for similar funded awards → identify programs and program officers → Grants.gov for open solicitations and deadlines. Typical places to check (program names change, so verify): NSF CBET (catalysis; separations/membranes), NSF DMR (solid-state materials chemistry), DOE BES (catalysis science; separation science), ACS PRF, ARPA-E for applied separations/energy.
5. **Collaborator search:** authors with recent papers on complementary methods (e.g. in-situ TEM, DFT of OSDA–framework interactions, membrane module testing) → filter by co-authorship distance and NSF/DOE awards.

**Privacy:** unpublished proposal ideas sent to a commercial API leave your machine. Check your institution's rules. Use a local model (Track B/C deployment) for the sensitive parts.

**Deliverable by week 4:** a working "proposal copilot" (prompted strong model + the tools above) that you're already using. Every question you ask it is saved, and good ones become Track B eval questions.

---

## 3. Track B: what to build for zeolite literature

### Zeolite-specific data sources (add to D1/D7 of the general plan)
| Source | What it gives |
|---|---|
| **IZA Database of Zeolite Structures** | 250+ framework type codes, topology, ring sizes, pore dimensions: a lookup tool and the source of many verifiable questions |
| **IZA "Verified Syntheses of Zeolitic Materials"** | expert-checked recipes, ideal gold data for synthesis extraction and QA |
| Literature-extracted zeolite synthesis datasets (e.g. Jensen et al., *ACS Cent. Sci.* 2019, built from thousands of papers) | gel compositions → products, a pretraining source for Track C models |
| Hypothetical zeolite and OSDA–zeolite datasets (computational) | a computational angle: OSDA binding / templating energies |
| Zeo++ / RASPA (software) | pore geometry and adsorption calculations as tools |
| Your group's publications + theses | the highest-relevance part of the corpus; also great eval-question material |

### Zeolite extraction schema (replaces `catalyst_record_v1`)
```json
{"framework": "MFI", "material_name": "2D-MFI nanosheets",
 "gel_molar_composition": {"SiO2": 1, "Al2O3": 0.0, "OSDA": 0.1, "NaOH": 0.0, "H2O": 80},
 "si_source": "TEOS", "al_source": null, "osda": "TPAOH", "mineralizer": "OH-",
 "seeds": {"used": true, "type": "MFI nanosheet", "wt_pct": 1.0},
 "aging": {"T_C": 25, "time_h": 12}, "crystallization": {"T_C": 150, "time_h": 72, "mode": "static"},
 "product_phase": "MFI", "phase_purity": "pure", "si_al_product": null,
 "morphology": "nanosheet", "thickness_nm": 5, "crystal_size_um": null,
 "post_treatment": ["calcination 550 C"],
 "application": {"type": "membrane", "separation": "p-xylene/o-xylene", "separation_factor": null},
 "evidence": {"doi": "10.xxxx/yyyy", "loc": "Experimental section"}}
```
*(Example values are placeholders; real records come from papers.)*

### Calculation generators (C2) for zeolite work
1. **Gel recipe ↔ molar ratios:** given SiO₂:Al₂O₃:OSDA:NaOH:H₂O ratios, batch size and reagents (TEOS, fumed silica, sodium aluminate, 40 wt% TPAOH…), compute reagent masses, or the reverse.
2. **Si/Al and SAR** from ICP results (wt% Si, Al); Si/Al ↔ SiO₂/Al₂O₃ conversions.
3. **Yield** (solid yield on an oxide basis) from product mass and gel composition.
4. **Scherrer crystallite size** from XRD peak width (with instrument broadening).
5. **Relative crystallinity** from XRD peak areas vs a reference.
6. **BET / t-plot:** micropore volume from given isotherm points.
7. **Membrane performance:** permeance (mol m⁻² s⁻¹ Pa⁻¹ ↔ GPU), separation factor vs ideal selectivity from feed/permeate compositions.
8. **Unit-cell / framework density** from cell parameters and the IZA T-atom count.
9. **Diffusion:** Thiele modulus / effectiveness factor for a zeolite crystal of given size (catalysis applications).

Each is a Python function → unlimited, correctly labelled training tasks (see section 5.2 of the general plan).

### Example eval questions (write your own, about 30 to start)
- "In doi:X, what crystallization temperature and time gave twin-free MFI nanosheets?" *(numeric, cited)*
- "Which framework codes have 10-membered-ring channels in 3D?" *(IZA lookup)*
- "Compute TPAOH (40 wt%) and TEOS masses for a 30 g gel with 1 SiO₂ : 0.25 TPAOH : 30 H₂O." *(calculation)*
- "Name two reported OSDA-free routes to MFI nanosheets and their key synthesis variable." *(entity list, cited)*

---

## 4. Library and access checklist (answer to "what should I collect?")

Collect these and the corpus plan can be finalized:

1. **Where you read papers.** List the 10–20 journals you read most. For zeolites these are probably *Microporous and Mesoporous Materials*, *Journal of Membrane Science* (Elsevier); *Chemistry of Materials*, *JACS*, *ACS Catalysis*, *ACS Applied Materials & Interfaces* (ACS); *Angewandte Chemie*, *AIChE Journal*, *Advanced Materials* (Wiley); *Science*, *Nature*, *Nature Materials*, *Science Advances*; and RSC journals. Which ones open with your university login?
2. **Subscribed databases:** Web of Science, Scopus, SciFinder-n (CAS), Reaxys? (Your library's A–Z database list shows this.)
3. **Text-and-data-mining (TDM) rights.** Email your subject librarian:
   > "I'm building a private research tool that searches the full text of papers in my field. Which of our publisher licences (Elsevier, ACS, Wiley, Springer Nature, RSC) allow text and data mining, and how do I get API access (e.g. an Elsevier API key, Wiley TDM token, Springer Nature TDM API)? Do they allow storing full texts locally for my own research?"
4. **Your existing PDF library:** a Zotero/EndNote/Mendeley export (BibTeX/RIS + PDFs). This is often the fastest legal start: papers you already downloaded for your own research.
5. **Your group's papers and theses:** a DOI list + thesis PDFs (institutional repository).
6. **Data policy:** any rules about sending unpublished data or proposals to cloud AI services.

Until TDM is confirmed, build the corpus from **abstracts (OpenAlex/Semantic Scholar) + open-access full texts + your own PDF library**. That's enough to start Track B.

---

## 5. Models: roles, local deployment, and your experiment data

### Which model is which
| Name | What it is | Used when | Example |
|---|---|---|---|
| **Policy model** | the model you **train**; it makes the decisions (which tool, what query, the answer). "Policy" is the RL word for "the thing that acts". | training, then **deployment**: this is your local model | Qwen3 ~8B (or a 30B MoE) |
| **Teacher model** | a larger, stronger helper used only **during development**: it writes example agent sessions for the supervised warm-up, or scores tokens for distillation (OPD) | development only, never deployed | a large open-weight model on Tinker (same tokenizer family as the policy, required for OPD) or a commercial API (check its terms on training with outputs) |
| **Judge model** | grades open-ended answers (Track A critique quality, reports) | evaluation / later RL | strong model + your rubric |
| **Surrogate model** (Track C) | a small numeric ML model trained on your experiments (Gaussian process, random forest) | called by the agent as a tool | "P(MFI pure) for this gel = 0.82" |

So the **local small-to-medium model you want is the policy model**. The teacher is a temporary helper for creating training examples.

### Deploying locally
- Train LoRA adapters on Tinker → `tinker checkpoint download` (or `tinker checkpoint push-hf` to a private Hugging Face repo) → serve locally with Ollama / LM Studio / llama.cpp (easy) or vLLM (fast, GPU).
- Rough hardware for 4-bit quantized models (approximate):

| Model size | Memory needed | Runs on |
|---|---|---|
| 4–8B | ~4–6 GB | Mac with 16 GB, or an RTX 3060/4060 laptop/desktop |
| 14B | ~9–10 GB | 16–24 GB Mac or a 12–16 GB GPU |
| 30B MoE (≈3B active) | ~18–20 GB | 32 GB Mac or a 24 GB GPU (RTX 3090/4090) |

- Expectation: a trained 8B model can beat a general model of the same size on *your* narrow tasks (zeolite QA, calculations, extraction). For open-ended Track A work (critique, novel ideas), frontier models will still be clearly better. Use the local model where privacy matters.

### Your experiment data: how many data points?
Your experiments serve **three different purposes**, each with its own threshold:

| Purpose | What "a data point" is | How many | Notes |
|---|---|---|---|
| **1. Searchable lab database** (agent answers "have we tried X?") | one synthesis record (gel, conditions, outcome) in the schema of section 3 | **any number; start with all historical records** (even 30–50 helps) | Value from day one. Include failures: they're the most useful records. |
| **2. Surrogate model** for phase/outcome prediction | record + measured outcome (phase, purity, crystallinity, Si/Al, thickness…) | **~50–150** for one framework system with 3–5 varied variables; **300–1,000+** across several frameworks/OSDAs | Fewer are needed if you combine with literature data (Jensen-style datasets) and only fine-tune on yours. Every failed or impure product counts as data. |
| **3. Bayesian optimization** (agent proposes the next experiments) | a new experiment you run | **~10–20 to start**, then batches of 3–8 per round | The surrogate improves with each round; this is "self-driving lab"-style. |
| (LLM fine-tuning itself) | an example *conversation/agent session*, not a data point | 500–3,000 sessions (SFT) + 1,000+ verifiable tasks (RL) | Built from literature and generators (Track B), not from raw experiment rows |

**Start recording now**, in a consistent spreadsheet with the schema fields above. Use one row per synthesis (including failed ones) and link each row to its XRD/SEM files. This is the single most useful preparation for Track C.

---

## 6. Revised timeline (part-time)

| Weeks | Track A (proposal copilot) | Track B (trained literature agent) | Track C (lab data) |
|---|---|---|---|
| 1–2 | connect OpenAlex, Semantic Scholar, NSF awards, PatentsView; first novelty check on your proposal | study guide Phases 0–2; GRPO runs on Tinker | design the experiment spreadsheet schema |
| 3–4 | funding + collaborator workflows; use it on the real proposal | corpus v1 (abstracts + OA + your PDFs); IZA lookup tool | enter historical experiments |
| 5–8 | maintenance; log questions → eval pool | tools + prompted baseline; 9 zeolite calculation generators; 150-question eval set | lab DB search tool for the agent |
| 9–12 | — | SFT warm start; extraction on 300+ papers (IZA recipes as gold) | first surrogate model (if ≥ 50 records) |
| 13–17 | swap in the trained local model where privacy matters | GRPO on Tinker; ablations | BO tool proposes experiments; you run 1–2 rounds |
| 18–20 | — | final eval; download the adapter; local deployment | agent combines literature + lab DB + surrogate |

**Week-1 to-do list**
1. Email your librarian (section 4.3).
2. Export your Zotero/PDF library and your group's DOI list.
3. Start the experiment spreadsheet (section 5).
4. Write 30 questions you actually asked (or wished you could ask) in the past month.
5. Share one aim from your proposal (you can redact it) to try the novelty-check workflow.
