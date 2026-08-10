"""Search compatibility helpers for the project-local UserBench patch.

The original judge treated a search request as an exact restatement of the
ground-truth task arguments.  TravelGym tasks, however, keep immutable route
and date arguments separate from elicitable preferences.  This module accepts
the immutable base arguments plus relevant preference qualifiers and leaves
the visible candidate list available for the Actor to filter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


_ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "flight": ("flight", "airline", "carrier"),
    "hotel": ("hotel", "lodging", "accommodation"),
    "apartment": ("apartment", "flat", "lodging"),
    # Do not treat a bare "vehicle" as a rental-car aspect: it commonly
    # appears inside apartment/flight preference text (for example,
    # "electric vehicle charging").  Keep only explicit rental phrases.
    "rental_car": ("rental car", "car rental", "rental vehicle"),
    "restaurant": ("restaurant", "dining", "eatery"),
}


def normalize_search_text(value: Any) -> str:
    """Normalize natural-language search text for explainable comparisons."""

    text = str(value).casefold()
    # Treat ordinal dates (10th/21st) as the corresponding cardinal number.
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _contains_value(request: str, value: Any) -> bool:
    expected = normalize_search_text(value)
    if not expected:
        return True
    return expected in request


def _mentions_aspect(request: str, aspect: str) -> bool:
    return any(
        normalize_search_text(alias) in request
        for alias in _ASPECT_ALIASES.get(aspect, (aspect,))
    )


def match_base_search_aspect(
    agent_request: str,
    task: Mapping[str, Any],
) -> str | None:
    """Return the unique aspect whose base arguments occur in a request.

    Preference qualifiers are intentionally ignored here.  The comparison is
    deliberately textual and deterministic: a query must mention one aspect
    and every immutable argument for that aspect.  If several aspects match,
    the request is considered unfocused and is left to the legacy judge.
    """

    request = normalize_search_text(agent_request)
    arguments = task.get("arguments")
    dimensions = task.get("dimensions")
    if not isinstance(arguments, Mapping) or not isinstance(dimensions, Sequence):
        return None

    matches: list[str] = []
    for raw_aspect in dimensions:
        aspect = str(raw_aspect).strip().casefold()
        if not aspect or not _mentions_aspect(request, aspect):
            continue
        aspect_arguments = arguments.get(aspect)
        if not isinstance(aspect_arguments, Mapping) or not aspect_arguments:
            continue
        if all(_contains_value(request, value) for value in aspect_arguments.values()):
            matches.append(aspect)

    return matches[0] if len(matches) == 1 else None


def deterministic_search_judgement(
    agent_request: str,
    task: Mapping[str, Any],
) -> dict[str, str] | None:
    """Build a positive judge result for base arguments plus preferences."""

    aspect = match_base_search_aspect(agent_request, task)
    if aspect is None:
        return None
    return {
        "alignment_judgement": "True",
        "alignment_aspect": aspect,
        "alignment_mode": "base_plus_preferences",
    }


def render_candidate_feedback(
    task: Mapping[str, Any],
    aspect: str,
    option_schemas: Mapping[str, str],
    *,
    refinement: bool = False,
) -> str:
    """Render a stable, actor-visible candidate list for a valid search."""

    options_by_aspect = task.get("all_options")
    options = (
        options_by_aspect.get(aspect, [])
        if isinstance(options_by_aspect, Mapping)
        else []
    )
    schema = str(option_schemas.get(aspect, "")).strip()
    if refinement:
        prefix = (
            "The base task search is already complete for "
            f"<{aspect}>. Relevant preference qualifiers are accepted; "
            "use the existing visible candidates to filter by preference."
        )
    else:
        prefix = (
            "The base task search arguments are correct. Relevant preference "
            "qualifiers may be included and do not invalidate the search."
        )
    lines = [prefix]
    if schema:
        lines.extend(["", schema])
    lines.append(f"Here are all the options for <{aspect}>:")
    lines.extend(json.dumps(option, ensure_ascii=False) for option in options)
    return "\n".join(lines).strip()
