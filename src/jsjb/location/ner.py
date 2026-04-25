from __future__ import annotations

# Public location recognition entry point.
# The current implementation lives in location_ner_rule_based.py.
from src.jsjb.location.rule_based import (  # noqa: F401
    BeijingDistrictResolver,
    DISTRICT_NAMES,
    LocationHit,
    LocationNER,
)

BEIJING_DISTRICTS = set(DISTRICT_NAMES)
