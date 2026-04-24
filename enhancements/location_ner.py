from __future__ import annotations

# Compatibility shim for modules that still import enhancements.location_ner.
# The current implementation lives in location_ner_rule_based.py.
from enhancements.location_ner_rule_based import (  # noqa: F401
    BeijingDistrictResolver,
    DISTRICT_NAMES,
    LocationHit,
    LocationNER,
)

BEIJING_DISTRICTS = set(DISTRICT_NAMES)
