from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


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
    headlines: Sequence[Headline],
    active_domains: Iterable[str],
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


def _resolve_concepts(active_domains: Iterable[str]) -> dict[str, list[str]]:
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


def _diversify(items: Sequence[ScoredHeadline]) -> list[ScoredHeadline]:
    seen: list[str] = []
    diversified: list[ScoredHeadline] = []
    for item in sorted(items, key=lambda entry: entry.score, reverse=True):
        normalized = _normalize_title(item.headline.title)
        if any(_is_similar(normalized, existing) for existing in seen):
            continue
        seen.append(normalized)
        diversified.append(item)
    return diversified


def _limit_sources(
    items: Sequence[ScoredHeadline],
    max_per_source: int,
) -> list[ScoredHeadline]:
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
