from __future__ import annotations

import csv
import html
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import URLError
from urllib.parse import parse_qs
from urllib.request import Request, urlopen
from xml.etree import ElementTree

APP_VERSION = "0.12.0"

@dataclass(frozen=True)
class Headline:
    title: str
    url: str | None = None
    source: str | None = None
    description: str | None = None
    time_ago: str | None = None


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

TAG_LIST = [
    "humanfactors",
    "ergonomics",
    "systemsengineering",
    "industrialdesign",
    "humanoperators",
    "occupationalhealth",
    "occupationalsafety",
    "usabilityengineering",
    "HCI",
    "cognitiveergonomics",
    "humanperformance",
    "industrialengineering",
    "workplacedesign",
    "cognitiveengineering",
    "taskanalysis",
    "anthropometrics",
    "riskassessment",
    "safetyengineering",
    "medicaldevices",
    "healthcareergonomics",
    "cyberphysicalsystems",
    "transportationsystems",
    "industrialprocess",
    "AccessibilityDesign",
    "BehavioralEconomics",
    "UXResearch",
    "HumanCenteredAI",
    "UserInterfaceDesign",
    "HealthcareUX",
    "SmartDevices",
    "WearableTech",
    "InclusiveDesign",
    "SustainableDesign",
]

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
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    if any("," in line for line in lines):
        return _parse_csv_lines(lines)
    return _parse_block_lines(lines)


def parse_rss_feeds(feed_urls: list[str], max_age_days: int) -> tuple[list[Headline], str]:
    headlines: list[Headline] = []
    attempts = 0
    parsed = 0
    cutoff = time.time() - max(0, max_age_days) * 86400
    for url in feed_urls:
        if not url:
            continue
        attempts += 1
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "HF-News-Triage-Tool/1.0 (+rss-fetch)",
                    "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
                },
            )
            with urlopen(request, timeout=6) as response:
                content = response.read()
        except (URLError, TimeoutError, ValueError):
            continue
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            continue
        parsed += 1
        headlines.extend(_parse_rss_root(root, url, cutoff))
    status = f"RSS: loaded {len(headlines)} items from {parsed}/{attempts} feeds (last {max(0, max_age_days)} days)."
    return headlines, status


def format_headlines_for_input(headlines: list[Headline]) -> str:
    lines: list[str] = []
    for item in headlines:
        if item.source:
            lines.append(item.source)
        lines.append(item.title)
        if item.description:
            lines.append(item.description)
        if item.time_ago:
            lines.append(item.time_ago)
        lines.append("")
    return "\n".join(lines).strip()


def _parse_rss_root(root: ElementTree.Element, feed_url: str, cutoff: float) -> list[Headline]:
    channel = _find_first_by_local(root, "channel")
    if channel is not None:
        return _parse_rss_items(channel, feed_url, cutoff)
    if _find_children_by_local(root, "entry"):
        return _parse_atom_entries(root, feed_url, cutoff)
    items = _find_children_by_local(root, "item")
    if items:
        fake_channel = ElementTree.Element("channel")
        title_node = ElementTree.SubElement(fake_channel, "title")
        title_node.text = feed_url
        for item in items:
            fake_channel.append(item)
        return _parse_rss_items(fake_channel, feed_url, cutoff)
    return []


def _parse_rss_items(
    channel: ElementTree.Element,
    feed_url: str,
    cutoff: float,
) -> list[Headline]:
    items: list[Headline] = []
    source = _get_text(_find_first_by_local(channel, "title")) or feed_url
    for item in _find_children_by_local(channel, "item"):
        title = _get_text(_find_first_by_local(item, "title"))
        if not title:
            continue
        link = _get_text(_find_first_by_local(item, "link"))
        description = _get_text(_find_first_by_local(item, "description"))
        if not description:
            description = _get_text(_find_first_by_local(item, "content"))
        published = _get_text(_find_first_by_local(item, "pubDate"))
        if not published:
            published = _get_text(_find_first_by_local(item, "published"))
        if cutoff and published:
            published_ts = _parse_rss_date(published)
            if published_ts and published_ts < cutoff:
                continue
        items.append(
            Headline(
                title=title,
                url=link,
                source=source,
                description=_trim_description(description),
                time_ago=_format_time_ago(published),
            )
        )
    return items


def _parse_atom_entries(
    root: ElementTree.Element,
    feed_url: str,
    cutoff: float,
) -> list[Headline]:
    items: list[Headline] = []
    source = _get_text(_find_first_by_local(root, "title")) or feed_url
    for entry in _find_children_by_local(root, "entry"):
        title = _get_text(_find_first_by_local(entry, "title"))
        if not title:
            continue
        link = None
        for link_node in _find_children_by_local(entry, "link"):
            if link_node.attrib.get("rel") in (None, "alternate"):
                link = link_node.attrib.get("href") or (link_node.text or "").strip() or None
                break
        description = _get_text(_find_first_by_local(entry, "summary")) or _get_text(
            _find_first_by_local(entry, "content")
        )
        published = _get_text(_find_first_by_local(entry, "updated")) or _get_text(
            _find_first_by_local(entry, "published")
        )
        if cutoff and published:
            published_ts = _parse_rss_date(published)
            if published_ts and published_ts < cutoff:
                continue
        items.append(
            Headline(
                title=title,
                url=link,
                source=source,
                description=_trim_description(description),
                time_ago=_format_time_ago(published),
            )
        )
    return items


def _parse_csv_lines(lines: list[str]) -> list[Headline]:
    headlines: list[Headline] = []
    for line in lines:
        if not line:
            continue
        try:
            parsed = next(csv.reader([line]))
        except csv.Error:
            parsed = [line]
        title = parsed[0].strip() if len(parsed) > 0 else ""
        if not title:
            continue
        url = parsed[1].strip() if len(parsed) > 1 and parsed[1].strip() else None
        source = parsed[2].strip() if len(parsed) > 2 and parsed[2].strip() else None
        description = parsed[3].strip() if len(parsed) > 3 and parsed[3].strip() else None
        time_ago = parsed[4].strip() if len(parsed) > 4 and parsed[4].strip() else None
        headlines.append(
            Headline(
                title=title,
                url=url,
                source=source,
                description=description,
                time_ago=time_ago,
            )
        )
    return headlines


def _parse_block_lines(lines: list[str]) -> list[Headline]:
    headlines: list[Headline] = []
    index = 0
    while index < len(lines):
        source = lines[index].strip()
        index += 1
        if not source:
            continue

        if index >= len(lines):
            break
        possible_title = lines[index].strip()
        if possible_title.isdigit():
            index += 1
            if index >= len(lines):
                break
            possible_title = lines[index].strip()
        title = possible_title
        index += 1
        if not title:
            continue

        description = None
        time_ago = None
        if index < len(lines) and lines[index].strip() == "•":
            index += 1
        if index < len(lines):
            description_candidate = lines[index].strip()
            if _looks_like_time(description_candidate):
                time_ago = description_candidate
                index += 1
            else:
                description = description_candidate
                index += 1
                if index < len(lines) and lines[index].strip() == "•":
                    index += 1
                    if index < len(lines):
                        description = lines[index].strip() or description
                        index += 1
                if index < len(lines) and _looks_like_time(lines[index].strip()):
                    time_ago = lines[index].strip()
                    index += 1

        headlines.append(
            Headline(
                title=title,
                source=source,
                description=description,
                time_ago=time_ago,
            )
        )
    return headlines


def _looks_like_time(value: str) -> bool:
    return bool(re.match(r"^\\d+\\s*[hm]$", value.strip().lower()))


def _get_text(node: ElementTree.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find_children_by_local(node: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in list(node) if _local_name(child.tag) == name]


def _find_first_by_local(node: ElementTree.Element, name: str) -> ElementTree.Element | None:
    for child in node.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _parse_rss_date(value: str) -> float | None:
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        pass

    cleaned = value.strip().replace("Z", "+00:00")
    for candidate in (
        cleaned,
        re.sub(r"\.(\d+)", "", cleaned),
    ):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _format_time_ago(value: str | None) -> str | None:
    if not value:
        return None
    timestamp = _parse_rss_date(value)
    if not timestamp:
        return None
    delta = int(time.time() - timestamp)
    hours = max(1, delta // 3600)
    if hours < 24:
        return f"{hours}h"
    days = max(1, hours // 24)
    return f"{days}d"


def _trim_description(description: str | None, limit: int = 220) -> str | None:
    if not description:
        return None
    cleaned = re.sub(r"<[^>]+>", "", description).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _llm_infer_score(headline: Headline) -> tuple[int, str | None]:
    prompt = (
        "You are an assistant scoring news headlines for human factors relevance. "
        "Given a headline and description, return JSON with keys score (1-10) and "
        "summary (short phrase). Focus on HF/UX/HCI/safety/automation relevance, "
        "and allow broad applicability for a human factors podcast.\\n\\n"
        f"Headline: {headline.title}\\n"
        f"Description: {headline.description or ''}\\n"
        "JSON:"
    )
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": 120,
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = Request(
        "http://localhost:8080/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return 0, None

    text = data.get("content") or data.get("text") or ""
    match = re.search(r"\\{.*\\}", text, re.DOTALL)
    if not match:
        return 0, None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0, None
    score = parsed.get("score")
    summary = parsed.get("summary")
    if isinstance(score, int) and 1 <= score <= 10:
        return score, str(summary).strip() if summary else None
    return 0, None


def score_headlines(
    headlines: list[Headline],
    active_domains: list[str],
    score_threshold: int,
    use_llm: bool,
    tag_list: list[str],
    top_n: int = 20,
    max_per_source: int = 3,
) -> list[ScoredHeadline]:
    active_concepts = _resolve_concepts(active_domains)
    scored = [
        _score_one(headline, active_concepts, use_llm=use_llm, tag_list=tag_list)
        for headline in headlines
    ]
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


def _score_one(
    headline: Headline,
    concepts: dict[str, list[str]],
    use_llm: bool,
    tag_list: list[str],
) -> ScoredHeadline:
    text = " ".join(
        part.lower()
        for part in [headline.title, headline.description]
        if part and part.strip()
    )
    matches: list[tuple[str, str]] = []
    for concept, keywords in concepts.items():
        for keyword in keywords:
            if keyword in text:
                matches.append((concept, keyword))
                break

    tag_matches = _match_tags(text, tag_list)
    unique_concepts = {concept for concept, _ in matches}
    score = 1 + min(9, len(unique_concepts) * 2)
    tag_bonus = min(4, len(tag_matches))
    score = min(10, score + tag_bonus)
    llm_summary = None
    if use_llm:
        llm_score, llm_summary = _llm_infer_score(headline)
        llm_bonus = max(0, min(3, llm_score - score))
        score = min(10, score + llm_bonus)
    tags = sorted(unique_concepts | set(tag_matches))[:5]
    angle = _build_angle(tags)
    confidence = _confidence_for(score, len(unique_concepts))
    rationale = [f"Matched {concept}." for concept in tags] or [
        "No strong human factors cues found in the headline.",
    ]
    if tag_matches:
        rationale.append(f"Tag signal: {', '.join(tag_matches[:5])}.")
    if llm_summary:
        rationale.append(f"LLM signal: {llm_summary}")
    rationale.extend(_build_angle_prompts(tags))
    rationale.append(_build_topic_framing(tags))
    rationale.extend(
        _build_confidence_drivers(
            headline=headline,
            unique_concepts=len(unique_concepts),
            tag_matches=len(tag_matches),
            llm_used=use_llm,
        )
    )
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


def _build_angle_prompts(tags: list[str]) -> list[str]:
    prompts = []
    prompt_map = {
        "trust in automation": "How does this affect trust in automation?",
        "situation awareness": "What does this mean for situation awareness on the job?",
        "workload": "Does this shift workload, fatigue, or cognitive load?",
        "usability": "What usability friction or design trade-off is exposed?",
        "error": "Is there a clear human error mechanism or recovery path?",
        "safety culture": "What does this say about safety culture and oversight?",
        "decision making": "How are decisions supported or undermined?",
        "human-ai teams": "Where do humans and AI coordinate or clash?",
        "training": "What training gaps or performance aids show up?",
        "healthcare": "How does this impact clinical work and patient safety?",
        "transport": "What operational safety or automation issue is at play?",
        "org factors": "How do teams, handoffs, or coordination shift here?",
    }
    for tag in tags:
        prompt = prompt_map.get(tag)
        if prompt:
            prompts.append(f"Angle prompt: {prompt}")
    return prompts[:3]


def _build_topic_framing(tags: list[str]) -> str:
    if not tags:
        return "Show framing: This could be a broader story about human-system fit once more context is gathered."
    if len(tags) == 1:
        return f"Show framing: Use this as a case study in {tags[0]} and its real-world implications."
    return (
        "Show framing: Highlight how "
        + ", ".join(tags[:2])
        + " connect to everyday human performance and system design."
    )


def _build_confidence_drivers(
    headline: Headline,
    unique_concepts: int,
    tag_matches: int,
    llm_used: bool,
) -> list[str]:
    drivers = []
    if headline.description:
        drivers.append("Confidence driver: Description provides added context.")
    else:
        drivers.append("Confidence driver: Limited context (headline only).")
    if unique_concepts >= 2:
        drivers.append("Confidence driver: Multiple HF concepts detected.")
    else:
        drivers.append("Confidence driver: Few explicit HF concepts detected.")
    if tag_matches:
        drivers.append("Confidence driver: Tag list aligned with the story language.")
    if llm_used:
        drivers.append("Confidence driver: Local LLM signal included.")
    return drivers


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


def _match_tags(text: str, tag_list: list[str]) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for tag in tag_list:
        normalized = _normalize_tag(tag)
        if not normalized:
            continue
        if normalized in lowered.replace("-", " "):
            matched.append(tag)
            continue
        if _all_words_present(normalized, lowered):
            matched.append(tag)
    return matched


def _normalize_tag(tag: str) -> str:
    tag = tag.replace("_", " ").replace("-", " ")
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", tag)
    return re.sub(r"\\s+", " ", spaced).lower().strip()


def _all_words_present(normalized: str, text: str) -> bool:
    words = [word for word in normalized.split() if word]
    return all(word in text for word in words)


def _parse_custom_tags(raw_text: str) -> list[str]:
    tags: list[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if cleaned:
            tags.append(cleaned)
    return tags or TAG_LIST


def _parse_rss_urls(raw_text: str) -> list[str]:
    urls: list[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if cleaned:
            urls.append(cleaned)
    return urls


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
    use_llm: bool,
    tag_text: str,
    rss_text: str,
    rss_days: int,
    rss_status: str = "",
) -> str:
    escaped_headlines = html.escape(headlines_text)
    llm_checked = "checked" if use_llm else ""
    escaped_tags = html.escape(tag_text)
    escaped_rss = html.escape(rss_text)
    escaped_rss_status = html.escape(rss_status)
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
            description_html = (
                f"<p class=\"description\">{html.escape(item.headline.description)}</p>"
                if item.headline.description
                else ""
            )
            time_html = (
                f"<span class=\"time\">{html.escape(item.headline.time_ago)}</span>"
                if item.headline.time_ago
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
                      {description_html}
                    </div>
                    <div class=\"meta\">
                      {time_html}
                      <span class=\"confidence {confidence_lower}\">{confidence}</span>
                    </div>
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
                    description_html=description_html,
                    time_html=time_html,
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
        --bg: #f5f7fb;
        --panel: #ffffff;
        --text: #10223b;
        --muted: #4b5d73;
        --primary: #15375f;
        --primary-strong: #2d76a0;
        --accent: #448841;
        --border: #dbe4ef;
        --low: #cb462b;
        --medium: #2d76a0;
        --high: #448841;
        --surface: #f0f5fb;
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
        position: relative;
      }}

      .hero {{
        margin-bottom: 20px;
        padding: 20px;
        border-radius: 16px;
        background: linear-gradient(120deg, #15375f 0%, #2d76a0 100%);
        color: #ffffff;
        box-shadow: 0 12px 30px rgba(21, 55, 95, 0.25);
      }}

      .hero h1 {{
        margin: 0 0 6px;
        font-size: 1.8rem;
      }}

      .hero p {{
        margin: 0;
        color: #e6eef7;
      }}

      .panel {{
        background: var(--panel);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 10px 24px rgba(30, 31, 37, 0.08);
      }}

      .panel-header {{
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-bottom: 16px;
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

      .supporting {{
        font-size: 0.95rem;
        color: var(--muted);
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
        background: var(--surface);
      }}

      .controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 16px 24px;
        align-items: flex-start;
      }}

      .settings-toggle {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        font-weight: 600;
        color: var(--primary);
        border: 1px solid var(--border);
        padding: 8px 12px;
        border-radius: 999px;
        background: #ffffff;
        box-shadow: 0 6px 14px rgba(21, 55, 95, 0.08);
      }}

      .settings-toggle span {{
        display: inline-flex;
        flex-direction: column;
        gap: 4px;
      }}

      .settings-toggle span i {{
        width: 18px;
        height: 2px;
        background: var(--primary);
        border-radius: 999px;
      }}

      .settings-header {{
        display: flex;
        justify-content: flex-start;
        margin: 0 0 16px;
      }}

      .settings-floating {{
        position: fixed;
        top: 16px;
        left: 16px;
        z-index: 50;
      }}

      .settings-label {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }}

      .settings-panel {{
        display: none;
        margin-top: 12px;
        padding: 16px;
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid var(--border);
      }}

      #settings-toggle:checked ~ .settings-panel {{
        display: grid;
        gap: 12px;
      }}

      .settings-panel textarea {{
        min-height: 160px;
        font-size: 0.9rem;
      }}

      .settings-panel input[type=\"number\"] {{
        width: 120px;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: #ffffff;
      }}

      .settings-note {{
        font-size: 0.85rem;
        color: var(--muted);
      }}

      .controls > div {{
        min-width: 200px;
      }}

      .controls input[type=\"number\"] {{
        width: 88px;
        padding: 8px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: #ffffff;
      }}

      .domain-group {{
        display: grid;
        gap: 8px;
      }}

      .checkbox {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: var(--muted);
      }}

      .primary {{
        background: var(--primary);
        color: white;
        border: none;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 6px 14px rgba(21, 55, 95, 0.2);
      }}

      .secondary {{
        background: transparent;
        color: var(--primary);
        border: 1px solid var(--border);
        padding: 10px 16px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
      }}

      .form-actions {{
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
      }}

      .version {{
        margin-top: 16px;
        font-size: 0.8rem;
        color: var(--muted);
        text-align: right;
      }}

      .rss-status {{
        font-size: 0.85rem;
        color: var(--muted);
        margin: 6px 0 0;
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
        background: #ffffff;
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
        background: #eaf2fb;
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
        background: #eaf2fb;
        color: var(--primary);
        padding: 4px 8px;
        border-radius: 999px;
        font-size: 0.8rem;
      }}

      .meta {{
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 6px;
      }}

      .source {{
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 0.85rem;
      }}

      .description {{
        margin: 6px 0 0;
        color: var(--text);
        font-size: 0.9rem;
      }}

      .time {{
        font-size: 0.8rem;
        color: var(--muted);
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
      <div class=\"settings-floating\">
        <label class=\"settings-toggle\" for=\"settings-toggle\" aria-label=\"Open settings\">
          <span class=\"settings-label\">
            <span>
              <i></i>
              <i></i>
              <i></i>
            </span>
            Settings
          </span>
        </label>
      </div>
      <section class=\"panel\">
        <div class=\"hero\">
          <h1>Human Factors Cast — Headline Triage</h1>
          <p>Take a deeper look into the human element in our ever changing digital world.</p>
        </div>
        <header class=\"panel-header\">
          <h2>Score headlines for HF/UX relevance</h2>
          <p class=\"supporting\">
            Use this triage view to spotlight stories with meaningful human-system angles for
            the show. Keep the shortlist focused, diverse, and grounded in HF practice.
          </p>
        </header>
        <form method=\"post\" class=\"form\">
          <label for=\"headlines\">Headlines (CSV: title, url, source, description, time ago or blocks)</label>
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
            <label class=\"checkbox\">
              <input type=\"checkbox\" name=\"use_llm\" value=\"yes\" {llm_checked} />
              Use local LLM (http://localhost:8080/completion)
            </label>
          </div>

          <input type=\"checkbox\" id=\"settings-toggle\" hidden />
          <div class=\"settings-panel\">
            <label for=\"custom_tags\">HF tag list (one per line)</label>
            <textarea id=\"custom_tags\" name=\"custom_tags\">{escaped_tags}</textarea>
            <label for=\"rss_urls\">RSS feeds (one URL per line)</label>
            <textarea id=\"rss_urls\" name=\"rss_urls\" placeholder=\"https://example.com/rss\">{escaped_rss}</textarea>
            <label for=\"rss_days\">Only include items from the last (days)</label>
            <input id=\"rss_days\" name=\"rss_days\" type=\"number\" min=\"0\" value=\"{rss_days}\" />
            <p class=\"settings-note\">
              Edit tags to tailor scoring to your show’s focus. These tags add extra boosts
              alongside the core HF concepts.
            </p>
            <p class=\"rss-status\">{escaped_rss_status}</p>
          </div>

          <div class=\"form-actions\">
            <button type=\"submit\" name=\"action\" value=\"fetch_feeds\" class=\"secondary\">
              Fetch RSS feeds
            </button>
            <button type=\"submit\" name=\"action\" value=\"run_triage\" class=\"primary\">
              Run triage
            </button>
          </div>
        </form>
        <p class=\"version\">Version {APP_VERSION}</p>
      </section>

      <section class=\"panel results\">
        <header class=\"panel-header\">
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
        tag_text = "\n".join(TAG_LIST)
        page = _render_page("", [], 5, [], False, tag_text, "", 7, "")
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
        use_llm = data.get("use_llm", ["no"])[0] == "yes"
        tag_text = data.get("custom_tags", ["\n".join(TAG_LIST)])[0]
        tag_list = _parse_custom_tags(tag_text)
        rss_text = data.get("rss_urls", [""])[0]
        rss_urls = _parse_rss_urls(rss_text)
        try:
            rss_days = int(data.get("rss_days", ["7"])[0])
        except ValueError:
            rss_days = 7
        action = data.get("action", ["run_triage"])[0]

        headlines = parse_headlines(headlines_text)
        rss_items: list[Headline] = []
        rss_status = ""
        if rss_urls:
            rss_items, rss_status = parse_rss_feeds(rss_urls, rss_days)
            if action == "fetch_feeds":
                headlines_text = format_headlines_for_input(rss_items)
                headlines = rss_items
            else:
                headlines.extend(rss_items)
        results = score_headlines(
            headlines=headlines,
            active_domains=active_domains,
            score_threshold=score_threshold,
            use_llm=use_llm,
            tag_list=tag_list,
        )
        page = _render_page(
            headlines_text,
            results,
            score_threshold,
            active_domains,
            use_llm,
            tag_text,
            rss_text,
            rss_days,
            rss_status,
        )
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
