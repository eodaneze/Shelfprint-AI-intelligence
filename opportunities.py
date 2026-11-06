from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Optional

from app import DEFAULT_DELIVERY_EFFICIENCY, MODEL_VERSION as ENGINE_VERSION, assess_opportunity
from demand_estimation import estimate_daily_demand
from validation import Issue, ValidationError

MODEL_VERSION = "baseline-v1"
DEFAULT_WINDOW_DAYS = 14
MIN_OBS_DAYS = 3.0          # sparse-data fallback: request-rate denominator is floored at this
STALE_COUNT_DAYS = 7
STALE_CONFIDENCE_FACTOR = 0.8
RISK_TO_CONTRACT = {"critical": "high", "high": "high", "medium": "medium", "low": "low"}


def _dt(v) -> datetime:
    d = datetime.fromisoformat(v.replace("Z", "+00:00")) if isinstance(v, str) else v
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_opportunity(item, as_of, target_cover_days, lead_time_days, safety_stock, window_days, delivery_efficiency):
    """Returns ("ok", opportunity) | ("needs_data", info) | ("skipped", info)."""
    key = {"retailerId": item["retailerId"], "productId": item["productId"]}
    observed_at = _dt(item["observedAt"])
    if observed_at > as_of:
        return "skipped", {**key, "reason": "count_observed_after_asOf"}

    start = as_of - timedelta(days=window_days)
    counts = {_dt(c["observedAt"]): c["quantity"] for c in item.get("countHistory") or []
              if start <= _dt(c["observedAt"]) <= as_of}
    counts[observed_at] = item["onHandUnits"]  # the confirmed snapshot is authoritative
    counts = sorted(counts.items())

    raw_deliveries = item.get("deliveries")
    deliveries = None if raw_deliveries is None else [(_dt(d["deliveredAt"]), d["units"]) for d in raw_deliveries]
    est = estimate_daily_demand(counts, deliveries)

    signals = [s for s in item.get("signals") or [] if start <= _dt(s["observedAt"]) <= as_of]
    unmet_units = sum(s["quantity"] for s in signals)
    times = [t for t, _ in counts] + [_dt(s["observedAt"]) for s in signals]
    obs_days = min(float(window_days), (as_of - min(times)).total_seconds() / 86400)
    unmet_rate = unmet_units / max(obs_days, MIN_OBS_DAYS) if unmet_units else 0.0

    depletion = est["daily_demand"]
    daily = (depletion or 0.0) + unmet_rate
    if daily <= 0:
        return "needs_data", {**key, "reason": "no_observed_demand", "onHandUnits": item["onHandUnits"],
                              "stockedOut": item["onHandUnits"] == 0,
                              "hint": "Need >= 3 days of confirmed counts or at least one demand signal."}

    source = ("depletion+requests" if depletion is not None and unmet_units
              else "depletion" if depletion is not None else "requests_only")
    if depletion is not None:
        confidence = min(1.0, est["confidence"] + (0.1 if unmet_units else 0.0))
    else:
        confidence = min(0.4, 0.1 + 0.05 * len(signals))
    stale_days = (as_of - observed_at).total_seconds() / 86400
    extra_issues = []
    if stale_days > STALE_COUNT_DAYS:
        confidence *= STALE_CONFIDENCE_FACTOR
        extra_issues.append(Issue("observedAt", "warning", "stale_stock_count",
                                  f"Latest confirmed count is {stale_days:.0f} days old; stock may have changed.").to_dict())
    confidence = round(confidence, 2)

    record = {
        "retailer_id": item["retailerId"], "product_id": item["productId"],
        "current_stock": item["onHandUnits"], "daily_demand": round(daily, 3),
        "target_coverage_days": target_cover_days, "lead_time_days": lead_time_days,
        "safety_stock": safety_stock, "observed_unmet_units": unmet_units,
        "evidence_confidence": confidence, "order_multiple": 1,
    }
    try:
        a = assess_opportunity(record, delivery_efficiency)
    except ValidationError as exc:
        return "skipped", {**key, "reason": "invalid_input", "issues": [i.to_dict() for i in exc.issues]}

    upc = item.get("unitsPerCarton")
    units = a["recommended_units"]
    assumptions = [
        "Stock unit is individual saleable units.",
        "Daily demand = depletion rate between confirmed counts + unmet requests per observation day.",
        "Depletion is treated as sales; corrections and shrinkage cannot be separated from sales.",
        f"Window {window_days} days; request-rate denominator floored at {MIN_OBS_DAYS:.0f} days (sparse-data fallback).",
        "Estimated missed units are not verified lost sales.",
        f"Delivery efficiency fixed at {delivery_efficiency} until the route module supplies it.",
    ]
    if est["demand_is_lower_bound"]:
        assumptions.append("Stockout intervals were excluded, so served demand is a lower bound.")
    if source == "requests_only":
        assumptions.append("No usable count history: demand comes from customer requests only.")

    return "ok", {
        **key,
        "asOf": as_of.isoformat(),
        "onHandUnits": item["onHandUnits"],
        "estimatedDailyDemandUnits": round(daily, 3),
        "daysOfCover": a["days_of_cover"],
        "recommendedUnits": units,
        "recommendedCartons": ceil(units / upc) if upc and units is not None else None,
        "estimatedMissedUnits": a["recoverable_units_estimate"],
        "missedUnitsBreakdown": {"observedUnmetUnits": unmet_units,
                                 "inferredLeadTimeShortfallUnits": a["lead_time_shortfall_units"]},
        "stockoutRisk": RISK_TO_CONTRACT[a["stockout_risk"]],
        "riskDetail": a["stockout_risk"],
        "confidence": confidence,
        "confidenceLabel": a["confidence_label"],
        "score": a["priority_score"],
        "scoreBreakdown": a["priority_breakdown"],
        "modelVersion": f"{MODEL_VERSION}/{ENGINE_VERSION}",
        "demandSource": source,
        "demandEstimate": {k: est[k] for k in ("daily_demand", "observed_days", "intervals_used",
                                                "intervals_censored", "intervals_skipped_restock", "method")},
        "inputWindow": {"from": start.isoformat(), "to": as_of.isoformat(), "days": window_days},
        "evidence": {"confirmedReportIds": list(item.get("confirmedReportIds") or []),
                     "signalIds": [s["id"] for s in signals if s.get("id")]},
        "dataQuality": a["data_quality"]["issues"] + extra_issues,
        "assumptions": assumptions,
    }


def compute_opportunities(items, as_of: Optional[datetime] = None, target_cover_days=7, lead_time_days=0,
                          safety_stock=0, window_days=DEFAULT_WINDOW_DAYS,
                          delivery_efficiency=DEFAULT_DELIVERY_EFFICIENCY) -> dict:
    as_of = _dt(as_of) if as_of else datetime.now(timezone.utc)
    ranked, needs_data, skipped, seen = [], [], [], set()
    for item in items:
        k = (item["retailerId"], item["productId"])
        if k in seen:
            skipped.append({"retailerId": k[0], "productId": k[1], "reason": "duplicate_in_request"})
            continue
        seen.add(k)
        status, out = build_opportunity(item, as_of, target_cover_days, lead_time_days,
                                        safety_stock, window_days, delivery_efficiency)
        {"ok": ranked, "needs_data": needs_data, "skipped": skipped}[status].append(out)

    ranked.sort(key=lambda o: (-o["score"], o["retailerId"], o["productId"]))
    for rank, o in enumerate(ranked, 1):
        o["rank"] = rank
    return {"asOf": as_of.isoformat(), "modelVersion": f"{MODEL_VERSION}/{ENGINE_VERSION}",
            "items": ranked, "needsData": needs_data, "skipped": skipped}
