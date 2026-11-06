"""SHELFPRINT AI demand intelligence engine (pure functions, no I/O)."""
from math import ceil
from typing import Optional

from validation import Issue, ValidationError, validate_record

MODEL_VERSION = "rules-v0.2"

# Tunable assumptions - show these in the UI as "model assumptions".
HIGH_BUFFER_DAYS = 2
MEDIUM_BUFFER_DAYS = 5
RISK_URGENCY = {"critical": 100, "high": 80, "medium": 50, "low": 20, "unknown": 0}
RECOVERABLE_UNITS_CAP = 10          # units that map to a 100 demand score; calibrate on seed data
DEFAULT_DELIVERY_EFFICIENCY = 50    # placeholder until the route module supplies it

REASONS = {
    "critical": "Critical stock position. Immediate replenishment attention is recommended.",
    "high": "Stock is approaching the expected replenishment threshold.",
    "medium": "Stock should be monitored and replenishment planned.",
    "low": "Current stock position appears relatively stable.",
    "unknown": "Insufficient demand information to determine stockout risk.",
}


def validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _exact_cover(current_stock: int, daily_demand: Optional[float]) -> Optional[float]:
    """Unrounded on purpose: rounding before comparing to lead time can flip risk bands."""
    validate_non_negative("current_stock", current_stock)
    if daily_demand is None or daily_demand <= 0:
        return None
    validate_non_negative("daily_demand", daily_demand)
    return current_stock / daily_demand


def calculate_days_of_cover(current_stock: int, daily_demand: Optional[float]) -> Optional[float]:
    """Display value (2 decimals). The engine itself uses _exact_cover for risk decisions."""
    cover = _exact_cover(current_stock, daily_demand)
    return None if cover is None else round(cover, 2)


def classify_risk(current_stock: int, cover: Optional[float], lead_time_days: int) -> str:
    if current_stock == 0:
        return "critical"  # confirmed zero stock is critical even without a demand estimate
    if cover is None:
        return "unknown"
    if cover <= lead_time_days:
        return "critical"
    if cover <= lead_time_days + HIGH_BUFFER_DAYS:
        return "high"
    if cover <= lead_time_days + MEDIUM_BUFFER_DAYS:
        return "medium"
    return "low"


def calculate_recommended_units(
    current_stock: int,
    daily_demand: Optional[float],
    target_coverage_days: int,
    lead_time_days: int,
    safety_stock: int,
    order_multiple: int = 1,
) -> Optional[int]:
    for name, value in {
        "current_stock": current_stock,
        "target_coverage_days": target_coverage_days,
        "lead_time_days": lead_time_days,
        "safety_stock": safety_stock,
    }.items():
        validate_non_negative(name, value)
    if order_multiple <= 0:
        raise ValueError("order_multiple must be greater than zero.")
    if daily_demand is None or daily_demand <= 0:
        return None
    validate_non_negative("daily_demand", daily_demand)
    target_stock = daily_demand * (target_coverage_days + lead_time_days) + safety_stock
    # round() guards against float noise such as 4.000000000001 -> ceil 5
    raw_units = max(0, ceil(round(target_stock - current_stock, 6)))
    if raw_units == 0:
        return 0
    return ceil(raw_units / order_multiple) * order_multiple


def estimate_potential_missed_units(daily_demand, confirmed_stockout_days) -> Optional[int]:
    if daily_demand is None or confirmed_stockout_days is None:
        return None
    validate_non_negative("daily_demand", daily_demand)
    validate_non_negative("confirmed_stockout_days", confirmed_stockout_days)
    return ceil(round(daily_demand * confirmed_stockout_days, 6))


def estimate_lead_time_shortfall_units(daily_demand, cover, lead_time_days) -> Optional[int]:
    """Units expected to go unmet before a new delivery can arrive (forward-looking)."""
    if daily_demand is None or cover is None:
        return None
    return ceil(round(daily_demand * max(0.0, lead_time_days - cover), 6))


def confidence_label(confidence: float) -> str:
    return "HIGH" if confidence >= 0.7 else "MEDIUM" if confidence >= 0.4 else "LOW"


# Helpers from the original version, kept so existing tests and callers keep working.
def calculate_urgency_score(days_of_cover, lead_time_days):
    if days_of_cover is None:
        return 0
    if days_of_cover <= lead_time_days:
        return 100
    if days_of_cover <= lead_time_days + HIGH_BUFFER_DAYS:
        return 80
    if days_of_cover <= lead_time_days + MEDIUM_BUFFER_DAYS:
        return 50
    return 20


def calculate_evidence_score(evidence_confidence):
    return round(evidence_confidence * 100)


def calculate_delivery_score(delivery_efficiency):
    return max(0, min(delivery_efficiency, 100))


def score_opportunity(urgency, recoverable_demand, evidence_confidence, delivery_efficiency) -> float:
    """Inputs must be normalized to 0-100 before scoring."""
    components = {
        "urgency": urgency,
        "recoverable_demand": recoverable_demand,
        "evidence_confidence": evidence_confidence,
        "delivery_efficiency": delivery_efficiency,
    }
    for name, value in components.items():
        if not 0 <= value <= 100:
            raise ValueError(f"{name} must be between 0 and 100.")
    return round(
        0.35 * urgency + 0.30 * recoverable_demand + 0.20 * evidence_confidence + 0.15 * delivery_efficiency,
        2,
    )


def calculate_demand_score(recoverable_units: Optional[int]) -> float:
    if recoverable_units is None:
        return 0.0
    return round(min(100.0, recoverable_units / RECOVERABLE_UNITS_CAP * 100), 2)


def assess_retailer(record: dict) -> dict:
    """Demand intelligence for one retailer-product observation. Raises ValidationError on bad input."""
    issues = validate_record(record)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise ValidationError(errors)

    stock = record["current_stock"]
    demand = record.get("daily_demand")
    lead = record["lead_time_days"]
    confidence = record.get("evidence_confidence", 0.5)

    cover = _exact_cover(stock, demand)
    risk = classify_risk(stock, cover, lead)
    recommended = calculate_recommended_units(
        stock, demand, record["target_coverage_days"], lead,
        record["safety_stock"], record.get("order_multiple", 1),
    )
    missed = estimate_potential_missed_units(demand, record.get("confirmed_stockout_days"))
    shortfall = estimate_lead_time_shortfall_units(demand, cover, lead)
    observed = record.get("observed_unmet_units")  # observed unmet requests (distinct from inferred)
    parts = [u for u in (missed, shortfall, observed) if u is not None]

    return {
        "retailer_id": record["retailer_id"],
        "product_id": record["product_id"],
        "days_of_cover": None if cover is None else round(cover, 2),
        "stockout_risk": risk,
        "recommended_units": recommended,
        "potential_missed_units": missed,
        "lead_time_shortfall_units": shortfall,
        "observed_unmet_units": observed,
        "recoverable_units_estimate": sum(parts) if parts else None,
        "evidence_confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "data_quality": {
            "status": "warnings" if issues else "ok",
            "issues": [i.to_dict() for i in issues],
        },
        "notes": (
            "Potential missed units are estimates, not confirmed lost sales."
            if missed is not None
            else "Potential missed demand unavailable: missing stockout or demand data."
        ),
        "model_version": MODEL_VERSION,
    }


def assess_opportunity(record: dict, delivery_efficiency: float = DEFAULT_DELIVERY_EFFICIENCY) -> dict:
    assessment = assess_retailer(record)
    risk = assessment["stockout_risk"]

    urgency = RISK_URGENCY[risk]
    demand = calculate_demand_score(assessment["recoverable_units_estimate"])
    evidence = round(assessment["evidence_confidence"] * 100)
    delivery = max(0, min(delivery_efficiency, 100))

    assessment["priority_score"] = score_opportunity(urgency, demand, evidence, delivery)
    assessment["priority_breakdown"] = {
        "urgency": urgency,
        "recoverable_demand": demand,
        "evidence_confidence": evidence,
        "delivery_efficiency": delivery,
    }

    units = assessment["recommended_units"]
    if units is not None and units > 0:
        action = "replenish"
    elif risk == "unknown":
        action = "collect_data"
    else:
        action = "review"
    assessment["recommendation"] = {"action": action, "recommended_units": units, "reason": REASONS[risk]}
    return assessment


if __name__ == "__main__":
    import json
    with open("sample_data.json") as fh:
        rows = json.load(fh)
    ranked = sorted((assess_opportunity(r) for r in rows), key=lambda a: -a["priority_score"])
    print(json.dumps(ranked, indent=2))
