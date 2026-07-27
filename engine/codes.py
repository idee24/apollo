"""Human-readable labels for Model A's low-cardinality feature codes.

These are GTD *schema* taxonomies (what an input code means), not incident data —
they define the model's input space and are used only to describe a scenario back
to the user in plain language. High-cardinality codes (country, nationality,
target subtype) are intentionally left as numbers here; naming them would require
the licensed GTD label tables, which stay server-side.

Kept in sync with docs/feature_glossary.md.
"""

from __future__ import annotations

REGION: dict[int, str] = {
    1: "North America", 2: "Central America & Caribbean", 3: "South America",
    4: "East Asia", 5: "Southeast Asia", 6: "South Asia", 7: "Central Asia",
    8: "Western Europe", 9: "Eastern Europe", 10: "Middle East & North Africa",
    11: "Sub-Saharan Africa", 12: "Australasia & Oceania",
}

ATTACKTYPE: dict[int, str] = {
    1: "Assassination", 2: "Armed Assault", 3: "Bombing/Explosion", 4: "Hijacking",
    5: "Hostage Taking (Barricade)", 6: "Hostage Taking (Kidnapping)",
    7: "Facility/Infrastructure Attack", 8: "Unarmed Assault", 9: "Unknown",
}

WEAPTYPE: dict[int, str] = {
    1: "Biological", 2: "Chemical", 3: "Radiological", 4: "Nuclear", 5: "Firearms",
    6: "Explosives", 7: "Fake Weapons", 8: "Incendiary", 9: "Melee",
    10: "Vehicle (non-explosive)", 11: "Sabotage Equipment", 12: "Other", 13: "Unknown",
}

TARGTYPE: dict[int, str] = {
    1: "Business", 2: "Government (General)", 3: "Police", 4: "Military",
    5: "Abortion Related", 6: "Airports & Aircraft", 7: "Government (Diplomatic)",
    8: "Educational Institution", 9: "Food or Water Supply", 10: "Journalists & Media",
    11: "Maritime", 12: "NGO", 13: "Other", 14: "Private Citizens & Property",
    15: "Religious Figures/Institutions", 16: "Telecommunication",
    17: "Terrorists/Non-State Militia", 18: "Tourists", 19: "Transportation",
    20: "Unknown", 21: "Utilities", 22: "Violent Political Party",
}

SUICIDE: dict[int, str] = {0: "not a suicide attack", 1: "suicide attack"}

# field -> code table, for generic lookups
FIELD_LABELS: dict[str, dict[int, str]] = {
    "region": REGION,
    "attacktype1": ATTACKTYPE,
    "weaptype1": WEAPTYPE,
    "targtype1": TARGTYPE,
    "suicide": SUICIDE,
}


def label(field: str, code) -> str | None:
    """Human label for a field's code, or None if unknown/unmapped."""
    if code is None:
        return None
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    return FIELD_LABELS.get(field, {}).get(code)


def describe_scenario(features: dict) -> str:
    """One-sentence plain-language description of a scenario from its codes.

    Uses labels where available and falls back to raw codes otherwise. Purely
    descriptive — it introduces no outcome information and never touches the label.
    """
    parts: list[str] = []
    atk = label("attacktype1", features.get("attacktype1"))
    if atk:
        parts.append(f"a {atk.lower()}")
    wpn = label("weaptype1", features.get("weaptype1"))
    if wpn:
        parts.append(f"using {wpn.lower()}")
    tgt = label("targtype1", features.get("targtype1"))
    if tgt:
        parts.append(f"against {tgt.lower()}")
    reg = label("region", features.get("region"))
    if reg:
        parts.append(f"in {reg}")
    elif features.get("country") is not None:
        parts.append(f"in country code {features['country']}")
    if features.get("iyear") is not None:
        parts.append(f"({features['iyear']})")
    suicide = label("suicide", features.get("suicide"))
    tail = f" This is coded as a {suicide}." if suicide else ""
    if not parts:
        return "An incident scenario with unspecified characteristics."
    return "An incident described as " + " ".join(parts) + "." + tail
