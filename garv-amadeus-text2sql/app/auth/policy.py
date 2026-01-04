# app/auth/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class AccessPolicy:
    allowed_tables: Set[str]
    redacted_columns: Set[str]
    max_rows: int = 2000               # hard cap
    require_limit: bool = True
    default_time_window_days: int = 7  # if missing


# Example roles (you can expand)
ROLE_POLICIES: Dict[str, AccessPolicy] = {
    "analyst": AccessPolicy(
        allowed_tables={
            "gold_airport_kpi_hourly",
            "gold_delay_reason_daily",
            "gold_anomaly_scores",
            "presecurity_events",
            "checkin_events",
        },
        redacted_columns=set(),
    ),
    "ops_manager": AccessPolicy(
        allowed_tables={
            "gold_airport_kpi_hourly",
            "gold_delay_reason_daily",
            "gold_anomaly_scores",
        },
        redacted_columns=set(),
    ),
    "restricted": AccessPolicy(
        allowed_tables={"gold_airport_kpi_hourly"},
        redacted_columns={"customer_id", "passport_no", "email"},
    ),
}


def get_policy(role: str) -> AccessPolicy:
    return ROLE_POLICIES.get(role, ROLE_POLICIES["analyst"])