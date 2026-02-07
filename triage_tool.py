from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from urllib.parse import parse_qs


@dataclass(frozen=True)
class Headline:
    title: str
    url: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class ScoredHeadline:
    headline: Headline
    score: int
    tags: list[str]
    angle: str
    confidence: str
    rationale: list[str]


CONCEPTS = {
    "workload": ["workload", "fatigue", "burnout", "cognitive load", "overload"],
    "situation awareness": ["situation awareness", "situational awareness", "awareness"],
    "trust in automation": ["trust", "automation", "autopilot", "ai assistant"],
    "usability": ["usability", "user experience", "ux", "interface", "design"],
    "error": ["error", "mistake", "incident", "failure", "crash"],
    "safety culture": ["safety culture", "compliance", "procedures", "oversight"],
    "decision making": ["decision", "judgment", "choice", "triage"],
    "human-ai teams": ["human-ai", "human ai", "copilot", "assistant"],
    "training": ["training", "simulation", "drill"],
    "healthcare": ["nurse", "doctor", "hospital", "clinical", "patient"],
    "transport": ["aviation", "rail", "driver", "traffic", "pilot"],
    "org factors": ["handoff", "shift", "team", "coordination", "communication"],
}

DOMAIN_MAP = {
    "ux_hci": {"usability", "decision making"},
    "safety": {"error", "safety culture", "situation awareness"},
    "healthcare": {"healthcare", "workload", "training"},
    "transport": {"transport", "situation awareness", "trust in automation"},
    "org_ops": {"org factors", "training", "workload"},
    "human_ai": {"human-ai teams", "trust in automation"},
}

DOMAINS = [
    ("ux_hci", "UX / HCI"),
    ("safety", "Safety"),
    ("healthcare", "Healthcare"),
    ("transport", "Transport"),
    ("org_ops", "Org / Ops"),
    ("human_ai", "Human-AI Teams"),
]


def parse_headlines(raw_text: str) -> list[Headline]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return []

    headlines: list[Headline] = []
    for line in lines:
        if "," in line:
            try:
                parsed = next(csv.reader([line]))
            except csv.Error:
                parsed = [line]
            title = parsed[0].strip()
            url = parsed[1].strip() if len(parsed) > 1 and parsed[1].strip() else None
            source = parsed[2].strip() if len(parsed) > 2 and parsed[2].strip() else None
            headlines.append(Headline(title=title, url=url, source=source))
        else:
            headlines.append(Headline(title=line))
    return headlines


def score_headlines(
    headlines: list[Headline],
    active_domains: list[str],
    score_threshold: int,
    top_n: int = 20,
    max_per_source: int = 3,
) -> list[ScoredHeadline]:
    active_concepts = _resolve_concepts(active_domains)
    scored = [_score_one(headline, active_concepts) for headline in headlines]
    filtered = [item for item in scored if item.score >= score_threshold]
    diversified = _diversify(filtered)
    sorted_items = sorted(
        diversified,
        key=lambda item: (item.score, _confidence_rank(item.confidence)),
        reverse=True,
    )
    limited = _limit_sources(sorted_items, max_per_source)
    return limited[:top_n]


def _resolve_concepts(active_domains: list[str]) -> dict[str, list[str]]:
    domain_keys = set(active_domains)
    if not domain_keys:
        return CONCEPTS

    allowed: set[str] = set()
    for domain in domain_keys:
        allowed.update(DOMAIN_MAP.get(domain, set()))
    if not allowed:
        return CONCEPTS
    return {concept: CONCEPTS[concept] for concept in allowed}


def _score_one(headline: Headline, concepts: dict[str, list[str]]) -> ScoredHeadline:
    text = headline.title.lower()
    matches: list[tuple[str, str]] = []
    for concept, keywords in concepts.items():
        for keyword in keywords:
            if keyword in text:
                matches.append((concept, keyword))
                break

    unique_concepts = {concept for concept, _ in matches}
    score = 1 + min(9, len(unique_concepts) * 2)
    tags = sorted(unique_concepts)[:5]
    angle = _build_angle(tags)
    confidence = _confidence_for(score, len(unique_concepts))
    rationale = [f"Matched {concept}." for concept in tags] or [
        "No strong human factors cues found in the headline.",
    ]
    return ScoredHeadline(
        headline=headline,
        score=score,
        tags=tags,
        angle=angle,
        confidence=confidence,
        rationale=rationale,
    )


def _build_angle(tags: list[str]) -> str:
    if not tags:
        return "Headline appears weakly tied to human factors without more context."
    if len(tags) == 1:
        return f"Strong signal for {tags[0]} in a human-system context."
    return f"Touches on {', '.join(tags[:-1])}, and {tags[-1]} in human-system work."


def _confidence_for(score: int, matches: int) -> str:
    if score >= 8 and matches >= 3:
        return "High"
    if score >= 5 and matches >= 2:
        return "Medium"
    return "Low"


def _confidence_rank(confidence: str) -> int:
    return {"Low": 0, "Medium": 1, "High": 2}.get(confidence, 0)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _diversify(items: list[ScoredHeadline]) -> list[ScoredHeadline]:
    seen: list[str] = []
    diversified: list[ScoredHeadline] = []
    for item in sorted(items, key=lambda entry: entry.score, reverse=True):
        normalized = _normalize_title(item.headline.title)
        if any(_is_similar(normalized, existing) for existing in seen):
            continue
        seen.append(normalized)
        diversified.append(item)
    return diversified


def _limit_sources(items: list[ScoredHeadline], max_per_source: int) -> list[ScoredHeadline]:
    if max_per_source <= 0:
        return list(items)
    counts: dict[str, int] = {}
    limited: list[ScoredHeadline] = []
    for item in items:
        source = item.headline.source or "unknown"
        counts[source] = counts.get(source, 0) + 1
        if counts[source] <= max_per_source:
            limited.append(item)
    return limited


def _is_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_words = set(a.split())
    b_words = set(b.split())
    overlap = len(a_words & b_words) / max(1, len(a_words | b_words))
    return overlap >= 0.7


def _render_page(
    headlines_text: str,
    results: list[ScoredHeadline],
    score_threshold: int,
    active_domains: list[str],
) -> str:
    escaped_headlines = html.escape(headlines_text)
    domain_markup = "".join(
        """
        <label class=\"checkbox\">
          <input type=\"checkbox\" name=\"domains\" value=\"{value}\" {checked} />
          {label}
        </label>
        """.format(
            value=value,
            label=label,
            checked="checked" if value in active_domains else "",
        )
        for value, label in DOMAINS
    )

    if results:
        cards = []
        for item in results:
            tags = "".join(f"<span>{html.escape(tag)}</span>" for tag in item.tags)
            rationale = "".join(
                f"<li>{html.escape(reason)}</li>" for reason in item.rationale
            )
            url_html = (
                f"<a href=\"{html.escape(item.headline.url)}\" target=\"_blank\""
                " rel=\"noreferrer\">"
                f"{html.escape(item.headline.url)}</a>"
                if item.headline.url
                else ""
            )
            source_html = (
                f"<p class=\"source\">{html.escape(item.headline.source)}</p>"
                if item.headline.source
                else ""
            )
            cards.append(
                """
                <article class=\"result-card\">
                  <div class=\"result-header\">
                    <span class=\"score\">{score}</span>
                    <div>
                      <h3>{title}</h3>
                      {url_html}
                      {source_html}
                    </div>
                    <span class=\"confidence {confidence_lower}\">{confidence}</span>
                  </div>
                  <div class=\"tags\">{tags}</div>
                  <p class=\"angle\">{angle}</p>
                  <ul>{rationale}</ul>
                </article>
                """.format(
                    score=item.score,
                    title=html.escape(item.headline.title),
                    url_html=url_html,
                    source_html=source_html,
                    confidence=item.confidence,
                    confidence_lower=item.confidence.lower(),
                    tags=tags,
                    angle=html.escape(item.angle),
                    rationale=rationale,
                )
            )
        results_html = "".join(cards)
        results_section = f"<div class=\"results-table\">{results_html}</div>"
    else:
        results_section = "<p class=\"empty\">No results yet. Paste headlines to get started.</p>"

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>HF Headline Triage</title>
    <style>
      :root {{
        color-scheme: light;
        font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        --bg: #f7f8fb;
        --panel: #ffffff;
        --text: #1e1f25;
        --muted: #5b616e;
        --primary: #335cff;
        --border: #e3e6ef;
        --low: #f4b942;
        --medium: #ff8c42;
        --high: #2f9e44;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        background: var(--bg);
        color: var(--text);
      }}

      .layout {{
        display: grid;
        grid-template-columns: minmax(280px, 1fr) minmax(320px, 1.2fr);
        gap: 24px;
        padding: 32px;
      }}

      .panel {{
        background: var(--panel);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 24px rgba(30, 31, 37, 0.08);
      }}

      h1,
      h2,
      h3 {{
        margin-top: 0;
      }}

      .subtitle {{
        color: var(--muted);
        margin-top: 4px;
      }}

      .form {{
        display: flex;
        flex-direction: column;
        gap: 16px;
      }}

      textarea {{
        width: 100%;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        font-size: 0.95rem;
        resize: vertical;
      }}

      .controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px 24px;
        align-items: flex-start;
      }}

      .controls input[type=\"number\"] {{
        width: 88px;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid var(--border);
      }}

      .domain-group {{
        display: grid;
        gap: 8px;
      }}

      .checkbox {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}

      .primary {{
        background: var(--primary);
        color: white;
        border: none;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
      }}

      .results-table {{
        display: grid;
        gap: 16px;
        max-height: 75vh;
        overflow-y: auto;
        padding-right: 8px;
      }}

      .result-card {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        background: #fbfbfd;
      }}

      .result-header {{
        display: grid;
        grid-template-columns: auto 1fr auto;
        gap: 12px;
        align-items: center;
      }}

      .score {{
        font-size: 1.4rem;
        font-weight: 700;
        background: #eef1ff;
        color: var(--primary);
        padding: 4px 10px;
        border-radius: 10px;
      }}

      .confidence {{
        padding: 4px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
      }}

      .confidence.low {{
        background: var(--low);
      }}

      .confidence.medium {{
        background: var(--medium);
      }}

      .confidence.high {{
        background: var(--high);
      }}

      .tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 8px;
      }}

      .tags span {{
        background: #eef1ff;
        color: #1d2c6b;
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 0.8rem;
      }}

      .source {{
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 0.85rem;
      }}

      .angle {{
        margin: 8px 0;
        color: var(--muted);
      }}

      .empty {{
        color: var(--muted);
        font-style: italic;
      }}

      @media (max-width: 900px) {{
        .layout {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main class=\"layout\">
      <section class=\"panel\">
        <header>
          <h1>HF Headline Triage</h1>
          <p class=\"subtitle\">
            Paste a list of headlines and rank them by human factors relevance with tags,
            angles, and a top 20 shortlist.
          </p>
        </header>
        <form method=\"post\" class=\"form\">
          <label for=\"headlines\">Headlines (one per line or CSV: title, url, source)</label>
          <textarea id=\"headlines\" name=\"headlines\" rows=\"14\" required>{escaped_headlines}</textarea>

          <div class=\"controls\">
            <div>
              <label for=\"score_threshold\">Score threshold</label>
              <input
                id=\"score_threshold\"
                name=\"score_threshold\"
                type=\"number\"
                min=\"1\"
                max=\"10\"
                value=\"{score_threshold}\"
              />
            </div>
            <div class=\"domain-group\">
              <span>Domains</span>
              {domain_markup}
            </div>
          </div>

          <button type=\"submit\" class=\"primary\">Run triage</button>
        </form>
      </section>

      <section class=\"panel results\">
        <header>
          <h2>Top Results</h2>
          <p class=\"subtitle\">Ranked by score, confidence, and diversity.</p>
        </header>
        {results_section}
      </section>
    </main>
  </body>
</html>
"""


class TriageHandler(BaseHTTPRequestHandler):
    def _send_html(self, html_body: str) -> None:
        encoded = html_body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        page = _render_page("", [], 5, [])
        self._send_html(page)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        data = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        headlines_text = data.get("headlines", [""])[0]
        try:
            score_threshold = int(data.get("score_threshold", ["5"])[0])
        except ValueError:
            score_threshold = 5
        active_domains = data.get("domains", [])

        headlines = parse_headlines(headlines_text)
        results = score_headlines(
            headlines=headlines,
            active_domains=active_domains,
            score_threshold=score_threshold,
        )
        page = _render_page(headlines_text, results, score_threshold, active_domains)
        self._send_html(page)

    def log_message(self, format: str, *args: object) -> None:
        return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = HTTPServer((host, port), TriageHandler)
    print(f"HF Headline Triage running on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
