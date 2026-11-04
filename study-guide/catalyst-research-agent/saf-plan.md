# Extending the Agent to Sustainable Aviation Fuel (SAF)

A forward-looking plan for after the zeolite agent works (roughly weeks 20–32 of the [zeolite plan](./zeolite-plan.md)), plus a few cheap steps you can take now. It reuses the [expert-distillation pipeline](./expert-distillation.md) with a new expert list (`expert-distillation/experts.saf.example.toml`).

> **Check before relying on numbers.** Standards, certified pathways and policy targets change. Values below are for orientation; take spec limits from your licensed copy of the current ASTM standards and policy figures from the official sources in section 4.

---

## 1. Why SAF is a natural extension of zeolite work

Most SAF routes end with the same acid and bifunctional catalysis problems that zeolites solve: building C₈–C₁₆ chains (oligomerization), branching them so the fuel doesn't freeze (hydroisomerization), and cutting long chains without overcracking to gas (hydrocracking). Diffusion length, acid-site density and location, and metal–acid balance decide selectivity and lifetime. These are the variables that nanosheet, hierarchical and OSDA-free synthesis control.

---

## 2. Route map

| Route (ASTM D7566 annex) | Feedstock | Key catalytic steps | Typical catalysts | Where zeolite design matters | Metrics |
|---|---|---|---|---|---|
| **HEFA-SPK** (A2) | fats, oils, greases | hydrodeoxygenation (HDO) / decarboxylation → n-alkanes; **hydroisomerization + mild hydrocracking** | sulfided NiMo/CoMo or Ni for HDO; Pt or Pd on **SAPO-11, ZSM-22 (TON), ZSM-23 (MTT), ZSM-48**, Beta, USY or amorphous silica–alumina | 1D 10-ring pore-mouth/key-lock selectivity; crystal size and acid density control isomer vs cracking; hierarchical/nanosized 1D zeolites | iso/n ratio, mono- vs multi-branched, jet-range yield, freezing point, H₂ use |
| **FT-SPK** (A1), **FT-SPK/A** (A4) | syngas from biomass, waste or CO₂ + H₂ (e-fuels) | Fischer–Tropsch (Co, Fe) → waxes; **wax hydrocracking/isomerization**; (A4) aromatics via alkylation/aromatization | Co/Fe FT catalysts; Pt/ASA, Pt/USY or Pt/zeolite for hydrocracking; ZSM-5 for aromatics | metal–acid proximity, mesoporosity for heavy waxes, ZSM-5 morphology for aromatics | chain-growth α, C₅₊ selectivity, jet cut yield, aromatics content |
| **ATJ-SPK** (A5), **ATJ-SKA** (A8) | ethanol, isobutanol and other light alcohols | **dehydration** → olefins; **oligomerization** → C₈–C₁₆; hydrogenation; (A8) aromatics | γ-Al₂O₃, H-ZSM-5, SAPO-34 (dehydration); **Ni-exchanged zeolites (Ni-Beta, Ni-Y, Ni on mesoporous aluminosilicates)** for ethylene; H-ZSM-5, Beta, acidic resins for C₃–C₄ olefins | acid density and Ni siting; **nanosheet/hierarchical MFI or Beta for lifetime against coke**; pore size for branching; Ni leaching/sintering | C₈–C₁₆ selectivity, branching, lifetime (time on stream), regenerability |
| **Methanol-to-jet** (check certification status) | methanol (bio or e-methanol) | methanol-to-olefins/hydrocarbons, then oligomerization + hydrogenation | SAPO-34, ZSM-5 (MTO/MTH), then oligomerization catalysts | **MFI nanosheets have extended MTH lifetime** in published work; Al siting, crystal thickness | olefin selectivity, lifetime, jet yield |
| **SIP** (A3), **CHJ** (A6), **HC-HEFA** (A7), co-processing (ASTM D1655) | sugars → farnesene; oils (hydrothermal); algal oils; co-feed in refinery | hydrogenation, hydrothermal conversion, conventional refining | metal, hydrotreating catalysts | less zeolite-centric; include for completeness | blend limits, yield |
| **Power-to-liquid** (via FT or methanol) | CO₂ + green H₂ | RWGS + FT, or CO₂ → methanol → jet | Cu/ZnO/Al₂O₃, In₂O₃; then as above | bifunctional oxide/zeolite (e.g. OX-ZEO) systems | CO₂ conversion, jet selectivity, H₂ efficiency |

Blend limits differ by annex (most are certified up to 50 vol% in conventional jet; some lower). Check the current D7566 revision.

### Jet fuel properties the agent must reason about (orientation only)
| Property | Approximate limit (Jet A / Jet A-1) | Why catalysis controls it |
|---|---|---|
| Freezing point | ≤ −40 °C (Jet A), ≤ −47 °C (Jet A-1) | branching from hydroisomerization |
| Flash point | ≥ 38 °C | light-end cut, cracking selectivity |
| Density at 15 °C | 775–840 kg/m³ | paraffinic SAF is light; cycloalkanes/aromatics raise density |
| Aromatics | ≤ 25 vol%; D7566 blends need ≥ 8 vol% | seal swell; ATJ-SKA and FT-SPK/A add aromatics |
| Net heat of combustion | ≥ 42.8 MJ/kg | composition |
| Viscosity at −20 °C | ≤ 8.0 mm²/s | carbon number and branching |
| Distillation (T10, FBP) | T10 ≤ 205 °C, FBP ≤ 300 °C | carbon-number distribution |

---

## 3. What your zeolite expertise maps to

| Zeolite skill | SAF problem it addresses |
|---|---|
| MFI nanosheets, 2D zeolites | shorter diffusion paths: less secondary cracking, slower coking in oligomerization and MTH; longer lifetime |
| Direct / OSDA-free syntheses | cost and scale for catalysts used at refinery scale |
| Control of Si/Al, Al siting, external vs internal acid sites | acid density for oligomerization vs cracking; pore-mouth catalysis in hydroisomerization |
| 1D 10-ring frameworks (TON, MTT, AEL) and their crystal size | isomer selectivity and freezing point in HEFA and FT upgrading |
| Ni/metal incorporation and siting | Ni-zeolite ethylene oligomerization; metal–acid balance in bifunctional catalysts |
| Zeolite membranes | separations in the process: ethanol/water dehydration, H₂ recovery, olefin/paraffin, product fractionation (a direct carryover of the membrane work) |
| Characterization (XRD, TEM/AFM, physisorption, NMR) | the same evidence criteria in the rubric bank |

These overlaps are also proposal angles: "nanosheet 1D zeolites for selective hydroisomerization of HEFA paraffins" or "membrane-assisted ethanol-to-jet" combine your current skills with SAF.

---

## 4. Key data sources

| Need | Source | Access |
|---|---|---|
| Specs and qualification | ASTM D7566 (synthetic blends), D1655 (Jet A/A-1), D4054 (qualification of new fuels); DEF STAN 91-091; CAAFI (pathway status, fuel qualification overview) | ASTM via library; CAAFI website free |
| Fuel composition–property data | CRC World Fuel Sampling Program reports; AFRL fuel reports; NIST Chemistry WebBook / TRC data; published SAF prescreening datasets (e.g. from fuel-property groups); DOE Co-Optima fuel property database (check scope) | mostly free |
| Catalysis literature | OpenAlex topic queries ("hydroisomerization n-hexadecane", "ethylene oligomerization nickel zeolite", "alcohol to jet"), expert list in `experts.saf.example.toml` | free + library |
| Life-cycle assessment | Argonne **GREET**; **ICAO CORSIA** default life-cycle emission values and methodology documents (fossil jet baseline 89 gCO₂e/MJ); EU RED / ReFuelEU documents | free |
| Techno-economics | NREL and PNNL design reports and SOT reports; DOE BETO publications; IEA Bioenergy reports | free |
| Patents | PatentsView / Lens.org: ATJ, HEFA, FT upgrading assignees | free |
| Policy context | US SAF Grand Challenge targets; ReFuelEU Aviation blending mandates; US tax-credit rules (change frequently) | official sites; **verify current status** |
| Process data for tools | Cantera / NIST thermo for equilibria; group-contribution parameters (Joback, Constantinou–Gani) | open |

---

## 5. What carries over from the zeolite agent

| Component | Carries over as | Changes |
|---|---|---|
| `build_expert_corpus.py` + `rubric_tools.py` | as is | new config (`experts.saf.example.toml`), SAF topic terms |
| Retrieval index, `search_corpus` tool | as is | add SAF corpus + reports (NREL/PNNL, CORSIA, GREET docs) |
| Extraction schema pattern | new schema (below) | catalyst-test records instead of synthesis records |
| Calculation-generator framework | as is | new generators (section 7) |
| Reward code (numeric tolerance, entity match, citation check) | as is | units: gCO₂e/MJ, $/gal, LHV |
| Rubric bank | extend | new dimensions: `fuel_spec_compliance`, `lca_boundary_assumptions`, `tea_assumptions`, `catalyst_lifetime_regeneration`, `feedstock_availability` |
| Tournament judge, critique tasks | as is | SAF planted flaws (e.g. LCA without land-use change, TEA without catalyst replacement cost, "jet yield" counted over the wrong carbon range) |
| Track C (lab DB, surrogate, BO) | as is | variables become catalyst + process conditions (T, P, WHSV, H₂/feed) with jet selectivity and lifetime as outcomes |
| Trained policy (LoRA) | continue training | mix ~30 % zeolite tasks into SAF training so it doesn't forget |

### Extraction schema for SAF catalyst tests
```json
{"route": "ATJ", "step": "ethylene oligomerization",
 "catalyst": "Ni-Beta", "support_framework": "BEA", "si_al": 12.5, "metal": "Ni", "metal_wt_pct": 2.0,
 "morphology": "nanocrystals", "pretreatment": "N2 500 C",
 "conditions": {"T_C": 120, "P_bar": 30, "WHSV_h-1": 2.0, "feed": "ethylene", "reactor": "fixed bed"},
 "results": {"conversion_pct": null, "selectivity_C8_C16_pct": null, "carbon_number_dist": null,
             "branching_index": null, "TOS_h": null, "deactivation": null},
 "fuel_properties": {"freezing_point_C": null, "density_kg_m3": null},
 "evidence": {"doi": "10.xxxx/yyyy", "loc": "Table 2"}}
```
*(Placeholder values; real records come from papers.)*

---

## 6. New tools

| Tool | What it does | Implementation |
|---|---|---|
| `fuel_spec_check(props)` | compares a property set with the Jet A/A-1 and D7566 limits | table you enter from the licensed standard |
| `blend_properties(components, fractions)` | volume-weighted density, mixing rules for viscosity; flags nonlinear properties (freezing point, flash point) | simple rules first; ML models from published prescreening work later |
| `estimate_properties(smiles or formula)` | boiling point, density, LHV, freezing point estimates for single hydrocarbons | group contribution (Joback) + NIST lookup; report uncertainty |
| `asf_distribution(alpha)` | Anderson–Schulz–Flory product distribution and jet-range fraction | analytic |
| `oligomer_distribution(params)` | Schulz–Flory distribution for olefin oligomerization; C₈–C₁₆ fraction | analytic |
| `h2_demand(feed, pathway)` | H₂ use for HDO vs decarboxylation of a triglyceride profile | stoichiometry |
| `lca_ci(inputs)` | carbon intensity (gCO₂e/MJ) and reduction vs the 89 gCO₂e/MJ baseline | simple accounting; cite GREET/CORSIA for factors |
| `mfsp(capex, opex, yield, ...)` | minimum fuel selling price by discounted cash flow | standard DCF |
| `search_corpus(..., route=...)` | literature retrieval filtered by route | existing tool + metadata |

---

## 7. Calculation generators (verifiable RL tasks)

1. ASF: given α, mass fraction of C₈–C₁₆; or α from a reported distribution.
2. Oligomerization: carbon-number distribution and jet fraction from a chain-growth probability.
3. HEFA H₂ consumption: triglyceride composition + HDO/DCO split → H₂ per kg feed; theoretical alkane yield.
4. Ethanol-to-jet mass balance: ethanol → ethylene → oligomers → jet; theoretical maximum mass yield (dehydration loses water).
5. WHSV/LHSV ↔ residence time and catalyst mass.
6. Carbon intensity and percent reduction vs fossil baseline; effect of an added process energy term.
7. MFSP sensitivity: change catalyst lifetime or feedstock price, recompute the selling price.
8. Blend density and aromatics content for a SAF/Jet A blend; check the 8 vol% aromatics floor.
9. Turnover frequency / site-time yield for oligomerization per Ni site or per acid site.
10. Thiele modulus / effectiveness factor for a zeolite crystal (from the zeolite generators) applied to hydroisomerization; shows the effect of nanosheet thickness.

---

## 8. Example evaluation questions

- "Which ASTM D7566 annex covers alcohol-to-jet with aromatics, and what is its maximum blend ratio?" *(standard lookup, cited)*
- "For n-hexadecane hydroisomerization over Pt/ZSM-22 in doi:X, what isomer yield was reported at maximum conversion?" *(numeric, cited)*
- "Given α = 0.88, what mass fraction of FT product is C₈–C₁₆?" *(calculation)*
- "Compute H₂ consumption (kg/kg feed) for complete HDO of triolein." *(calculation)*
- "Name two reported strategies to slow coking in Ni-zeolite ethylene oligomerization, with references." *(entity list, cited)*
- "A paper reports 'jet selectivity' as C₅–C₂₀. Why is that misleading, and what should be reported instead?" *(critique; planted-flaw style)*
- "Critique: an LCA claims 85 % reduction for ATJ from corn ethanol without land-use change. What's missing?" *(rubric-judged)*

---

## 9. Timeline (after the zeolite agent, part-time)

| Weeks | Work |
|---|---|
| now (cheap) | harvest the SAF expert list with `build_expert_corpus.py`; write 20 SAF questions you'd actually ask; note SAF angles in your proposals |
| 21–23 | SAF corpus + reports indexed; new extraction schema; property and spec tools |
| 24–26 | 10 SAF generators; 100-question SAF eval set; SAF rubric dimensions + advisor/SAF-collaborator calibration |
| 27–30 | SFT + GRPO continuing from the zeolite adapter, mixed with zeolite tasks |
| 31–32 | eval; Track C on any SAF catalyst tests (oligomerization/hydroisomerization screening + BO) |

### Open questions to decide later
1. Which route first? Recommended: **ATJ oligomerization or HEFA hydroisomerization**, because both are zeolite-centric and connect directly to your synthesis skills.
2. Experimental or computational? Experimental testing needs a fixed-bed reactor and GC (ideally GC×GC for composition). Computational work can start from published data and the property tools above.
3. Collaborators: SAF work benefits from a fuel-property or LCA/TEA collaborator; the collaborator search in Track A applies directly.
