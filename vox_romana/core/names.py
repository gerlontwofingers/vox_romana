"""Shared curated-name utilities for post-architect cast rewriting.

Bootstrap agents tend to converge on a small cluster of names. These helpers
rewrite LLM-generated names through a curated pool immediately after architect
JSON is parsed and before any profile-builder calls run.

Two rules keep the rewrite safe:
  1. If the user's brief explicitly names a person, preserve that name.
  2. Otherwise assign a deterministic curated full name so tests stay stable.
"""

from __future__ import annotations

import hashlib
import random
import re


FEMALE_GIVEN_NAMES = (
    "Amina",
    "Bianca",
    "Carmen",
    "Dalia",
    "Farah",
    "Ines",
    "Joanna",
    "Leila",
    "Lucia",
    "Marisol",
    "Marta",
    "Naomi",
    "Nora",
    "Priya",
    "Rania",
    "Samira",
    "Talia",
    "Yasmin",
    "Zara",
    "Zoya",
)

MALE_GIVEN_NAMES = (
    "Adrian",
    "Amir",
    "Basil",
    "Darius",
    "Elias",
    "Farid",
    "Gabe",
    "Hugo",
    "Isaac",
    "Jamal",
    "Karim",
    "Leo",
    "Mateo",
    "Nikolai",
    "Omar",
    "Rafael",
    "Sami",
    "Tariq",
    "Victor",
    "Yusuf",
)

NEUTRAL_GIVEN_NAMES = (
    "Ari",
    "Casey",
    "Jordan",
    "Milan",
    "Noa",
    "Rowan",
    "Sasha",
)

SURNAMES = (
    "Abbasi",
    "Alvarez",
    "Banerjee",
    "Costa",
    "Dawson",
    "El-Sayed",
    "Ferreira",
    "Haddad",
    "Ibarra",
    "Ivanov",
    "Kerr",
    "Khatib",
    "Mendes",
    "Molina",
    "Navarro",
    "Okafor",
    "Petrov",
    "Quintero",
    "Rahman",
    "Romero",
    "Santos",
    "Valente",
    "Yilmaz",
    "Zoric",
)

HONORIFICS = {
    "dr",
    "doctor",
    "mr",
    "mrs",
    "ms",
    "mx",
    "prof",
    "professor",
    "sir",
    "lady",
}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _strip_honorifics(name: str) -> str:
    tokens = _normalize(name).split()
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)
    return " ".join(tokens)


def brief_explicitly_mentions_name(brief: str, name: str) -> bool:
    """Return True when the user's brief explicitly names this person."""
    if not brief or not name:
        return False

    normalized_brief = _normalize(brief)
    candidates = {_normalize(name)}
    stripped = _strip_honorifics(name)
    if stripped:
        candidates.add(stripped)

    for candidate in candidates:
        if candidate and re.search(rf"\b{re.escape(candidate)}\b", normalized_brief):
            return True
    return False


def _given_pool(gender: str | None) -> tuple[str, ...]:
    normalized = (gender or "").strip().lower()
    if normalized == "female":
        return FEMALE_GIVEN_NAMES
    if normalized == "male":
        return MALE_GIVEN_NAMES
    return FEMALE_GIVEN_NAMES + MALE_GIVEN_NAMES + NEUTRAL_GIVEN_NAMES


def _seeded_choice(pool: tuple[str, ...], seed_material: str) -> str:
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return rng.choice(pool)


def curate_display_name(
    current_name: str,
    *,
    brief: str,
    slot_key: str,
    used_names: set[str],
    gender: str | None = None,
) -> str:
    """Return a preserved-or-curated display name for one cast slot."""
    if not current_name:
        return current_name

    if brief_explicitly_mentions_name(brief, current_name):
        used_names.add(current_name)
        return current_name

    given_pool = _given_pool(gender)
    for attempt in range(256):
        given = _seeded_choice(
            given_pool,
            f"{brief}|{slot_key}|given|{current_name}|{attempt}",
        )
        surname = _seeded_choice(
            SURNAMES,
            f"{brief}|{slot_key}|surname|{current_name}|{attempt}",
        )
        candidate = f"{given} {surname}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate

    used_names.add(current_name)
    return current_name


def build_curated_name_map(
    slots: list[tuple[str, str | None, str]],
    *,
    brief: str,
) -> dict[str, str]:
    """Build an old-name -> curated-name mapping for one architect response."""
    used_names: set[str] = set()
    mapping: dict[str, str] = {}
    for current_name, gender, slot_key in slots:
        if not current_name:
            continue
        mapping[current_name] = curate_display_name(
            current_name,
            brief=brief,
            slot_key=slot_key,
            used_names=used_names,
            gender=gender,
        )
    return mapping


def remap_name(name: str | None, mapping: dict[str, str]) -> str | None:
    if name is None:
        return None
    return mapping.get(name, name)


def remap_names(names: list[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(name, name) for name in names]