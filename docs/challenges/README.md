# Sponsor challenge briefs — 6th Global AI Hackathon (Hack-Nation × MIT Club of NorCal × MIT Club of Germany)

Six sponsor tracks. Pick **one**, per `docs/24h_plan.md` Phase 0. Full text: `challenge-brief.pdf` in each folder below.

## 01 — [The Negotiator](01-elevenlabs-negotiator/challenge-brief.pdf) (ElevenLabs)
Voice agents that phone real businesses, gather itemized quotes, and negotiate — for any "phone-priced" market (moving companies, medical bills, car buying, contractor bids, freight, equipment rental). Three required modules: **Estimator** (voice or document intake into a structured job spec), **Caller** (parallel outbound calls against at least 3 distinct negotiation styles — real businesses, role-played humans, or built counter-agents), **Closer** (negotiates using competing quotes as leverage, returns a ranked, evidence-backed report). Must disclose it's an AI, never fabricate quotes, and every call must end in a structured outcome. Built on ElevenLabs Agents + Twilio/SIP.

## 02 — [The VC Brain](02-maschmeyer-vc-brain/challenge-brief.pdf) (Maschmeyer Group)
An AI-first venture capital operating system covering Sourcing → Screening → Diligence → Decision, aiming to issue a $100K investment decision within 24 hours of an application. Three architecture layers: **Memory** (ingests decks, GitHub, launches, interviews; houses a persistent "Founder Score"), **Intelligence** (a configurable Thesis Engine + three independent scoring axes — Founder / Market / Idea-vs-Market — never averaged), **Experience** (investor-facing dashboard and evidence-backed memos with a per-claim "Trust Score"). Sourcing (finding founders before they fundraise) carries the most weight and explicitly targets the cold-start case — a founder with no track record.

## 03 — [RealDoor](03-realpage-realdoor/challenge-brief.pdf) (RealPage)
A renter-side "application-readiness copilot" for affordable housing. Three stages: **Profile** (extract fields from synthetic pay stubs/benefit letters, human-confirmed), **Understand** (explain one frozen program's eligibility rules with citations and deterministic math — never states a renter "is eligible"), **Prepare** (flag missing/expired documents, produce a renter-controlled, downloadable packet). Hard constraint: the system **never decides, scores, ranks, or approves** — a qualified human always decides. Heavy emphasis on responsible-AI controls (no hidden proxies, prompt-injection resistance, WCAG 2.2 AA accessibility) that must be demonstrated live, not just claimed.

## 04 — [Data Legend](04-databricks-data-legend/challenge-brief.pdf) (Databricks)
A "Trust Layer" app for Indian healthcare: turns 10,000 messy, unstructured healthcare-facility records (India) into decisions an NGO/public-health planner can act on and defend. Choose **one** of four mission tracks — Facility Trust Desk, Medical Desert Planner, Referral Copilot, or Data Readiness Desk — and ship it end-to-end as a live Databricks App (Free Edition) using Agent Bricks, Genie, MLflow 3, Vector Search, and Lakebase. Every output must cite the source text it's grounded in, and the app must distinguish "no evidence" from "no data" honestly.

## 05 — [Foundation Models for Women's Hormonal Health](05-womens-hormonal-health/challenge-brief.pdf) (Hack-Nation / OpenAI)
Contribute one open, reusable building block toward AI infrastructure for women's hormonal health (PCOS, endometriosis, menopause, etc.) — not a full foundation model in a weekend. Three possible layers: **Data & Benchmark Infrastructure** (a standardized multimodal dataset or benchmark with documented splits), **AI Model Infrastructure** (a focused, reproducible, explainable model — e.g. hormone-level or menopause-stage prediction), or **Application Infrastructure** (a symptom tracker, digital hormone journal, or similar tool built on the above). Open licensing of datasets/models/code is explicitly part of the scoring criteria, not optional polish. Suggested data: mcPHASES (PhysioNet), NHANES (CDC).

## 06 — [Genome Firewall](06-genome-firewall/challenge-brief.pdf) (Hack-Nation / OpenAI)
A strictly defensive biosecurity tool: given a reconstructed bacterial genome (FASTA), predict which antibiotics are likely to work, likely to fail, or return a **no-call**, with a calibrated confidence score and cited supporting genes/mutations — faster than the 1–3 day standard lab turnaround. Three modules: **Genome Reader** (FASTA → AMR features, via AMRFinderPlus or an alternative), **Predictor** (per-antibiotic prediction, with a deterministic molecular-target gate and homology-based train/test de-duplication), **Decision Report** (a Streamlit/Gradio demo that always recommends confirming with standard lab testing). The system must never design, modify, or suggest changes to an organism — recommended baseline is per-antibiotic regularized logistic regression, not a large genomic language model.

## Choosing a track

| If your team leans toward... | Consider |
|---|---|
| LLM agents, tool use, structured extraction, product/UX | 02 (VC Brain), 03 (RealDoor), 04 (Data Legend) |
| Voice / audio, real-time agents, telephony | 01 (Negotiator) |
| CNN/genomics ML, calibration, bio data | 06 (Genome Firewall) |
| Open dataset/benchmark work, medical ML | 05 (Women's Hormonal Health) |

All six explicitly reward **honest uncertainty** (no-call / abstention / confidence calibration) over forced answers — keep that in mind when picking a track and designing the MVP.
