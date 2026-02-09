# Headline Triage Tool for Human Factors Relevance

## Quickstart
1. Run the app (single-file executable):
   ```bash
   python triage_tool.py
   ```
2. Open `http://localhost:8000` and paste headlines (CSV: title, URL, source, description, time ago or source/title/description/time blocks).

## One-line summary
You paste (or pull) hundreds of headlines, the tool scores each one for human factors relevance (1–10), then returns a ranked top 20 with short “HF angles” you can use for research notes or podcast prep.

## Problem
You can’t scan enough headlines fast, and generic news ranking doesn’t answer the questions you actually care about:

- Is this meaningfully related to human factors?
- If yes, what’s the HF angle?
- Which 20 items deserve my attention today?

## Target users
- Human factors practitioners doing daily/weekly monitoring for safety, UX, automation, health, transport, defense, and org/ops.
- Science comms creators (podcast/newsletter) who need fast story selection plus a defensible HF framing.
- Researchers/students building reading lists and discussion prompts.

## Goals
- Score hundreds to thousands of headlines in one pass.
- Produce a 1–10 human factors relevance score plus a short rationale.
- Return a top 20 that is not repetitive (topic variety) and not clickbait-driven.
- Let you tune “what counts” as human factors for your context (UX, safety, HCI, org factors, human-AI teams, etc.).

## Non-goals
- Writing full article summaries from paywalled content.
- Replacing your judgement on what is “important” or “true.”
- Generating a full literature review (this tool just triages candidates).

## Inputs

### Supported ingestion
- Paste a list of headlines (newline / CSV).
- RSS feeds (per-source toggles).
- News APIs (optional).
- Internal feeds (Slack channels, email digests) if you want it later.

### Per-headline fields (minimum viable)
- `title` (required).
- `url` (optional but recommended).
- `source`, `published_at` (optional).

### Optional enrichment
- Fetch article lede/preview text (first ~1–2 paragraphs) for better scoring accuracy.
- Named entity extraction (company/product/person) to help grouping.

## Output
For the top 20 items, show:

- Headline + source + timestamp.
- HF relevance score (1–10).
- HF tags (2–5): e.g., workload, situation awareness, trust in automation, usability, error, safety culture, decision-making.
- 1–2 sentence “HF angle” (what makes it HF-relevant).
- Confidence flag (High/Med/Low).
- “Why scored this way” (short bullet explanation, not a wall of text).
- Actions: Save, Export, Send to show outline, Open link.

## Scoring rubric (1–10)
A simple rubric keeps the system consistent and explainable.

- **10**: Directly about human performance/behavior plus design/operations implications (e.g., workload, situation awareness, error, HCI, safety events tied to human-system issues).
- **7–9**: Strong HF connection but framed through adjacent domains (AI tools, policy, product changes) with clear human interaction impact.
- **4–6**: Indirect connection; would need extra context to make it an HF story.
- **1–3**: Little to no HF connection (pure finance, celebrity, sports scores, etc.).

You can anchor this rubric to known HF constructs like situation awareness and trust in automation (common in human–automation research) and to structured taxonomies used in safety contexts (e.g., National Academies guidance).

## How the analysis works (recommended approach)

### Stage A — Fast HF candidate detection (cheap + fast)
- Create a set of “HF concept vectors” (short descriptions for ~50–150 HF concepts).
- Embed each headline (and optional lede) and compute similarity to the concept set.
- Output: `hf_candidate_score` plus top matched concepts.

This style of similarity scoring maps well to embedding-based semantic search workflows (e.g., OpenAI platform patterns).

### Stage B — Final scoring + explanation (slower, higher quality)
For the candidates (or all headlines if volume is small):

- Run a classifier that outputs:
  - `hf_score` (1–10)
  - `tags`
  - `rationale` (bullets)
  - `confidence`
- Add a diversity pass so the top 20 isn’t 15 versions of the same story (cluster near-duplicates and keep the best representative).

Clustering and stream grouping are common patterns for handling high-volume news lists (e.g., arXiv examples).

## Ranking logic for “Top 20”
Rank by:

- `hf_score` (primary)
- `confidence` (secondary)
- novelty/diversity (cluster-based)
- recency (optional toggle)

Controls:
- “More diverse” vs “Most HF-related.”
- “Only score ≥ X.”
- Source weighting (trust your list, not the algorithm).

## Implementation notes (current)
- CSV ingestion supports `title, url, source, description, time ago` per line.
- Block ingestion supports source/title/description/time blocks like the sample you shared.
- Results are de-duplicated by title similarity and capped per source to reduce overlaps.
- `triage_tool.py` is a single-file executable with embedded UI styles.
- Optional local LLM scoring calls `http://localhost:8080/completion` (compatible with llama.cpp server) when enabled.

## UI sketch (simple)

### Main screen
- Left: ingestion (paste / RSS / API), run button, settings.
- Right: results table with columns: Score, Headline, Tags, Source, Confidence, Actions.

### Detail drawer (on click)
- HF angle.
- Matched concepts (from Stage A).
- Short “why” bullets.
- Link + notes field.

## Customization
Let you define “human factors” for your workflow:

- Toggle domains: UX/HCI, safety, healthcare, transport, org/ops, human-AI teams.
- Import a taxonomy (CSV) of your preferred tags (including org-specific language).
- Add “watch terms” and “ignore terms.”
- Create “show formats”:
  - Top 20 for podcast.
  - Top 20 for safety review.
  - Top 20 for UX research scan.

## Quality checks (what you measure)
- **Precision@20**: of the top 20, how many you rate ≥7?
- **Inter-rater agreement**: you + a second HF reviewer on scores.
- **Stability**: same input list gives near-identical top picks.
- **Diversity score**: number of distinct clusters represented in top 20.

### Error analysis buckets
- False positives (buzzword “human” stories).
- False negatives (HF implied but not explicit in headline).

## Risks and mitigations
- Headline-only ambiguity → add optional lede fetch and show confidence drops when context is thin.
- Buzzword bait (“AI”, “safety”) → require a concrete human-system mechanism in the rationale.
- Over-trust in the score → show rubric + “why” bullets and keep manual reorder prominent.
- Copyright / paywalls → store only small snippets or metadata; default to headline+URL unless you have rights.

## MVP scope (buildable)

### MVP (2–4 weeks of dev time for a small team)
- Paste headlines → score → top 20.
- Tags + 1–2 sentence HF angle.
- Export CSV/JSON.
- Basic settings: domain toggles + score threshold.

### V1
- RSS ingestion + scheduling.
- Clustering for near-duplicates.
- “Podcast outline” export (segments + talking prompts).
- Team workspace + shared tag taxonomy.
