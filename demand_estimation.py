from datetime import datetime
from typing import Optional, Sequence, Tuple

MIN_OBSERVED_DAYS = 3.0
FULL_CONFIDENCE_DAYS = 14.0


def estimate_daily_demand(
    observations: Sequence[Tuple[datetime, int]],
    deliveries: Optional[Sequence[Tuple[datetime, int]]] = None,
) -> dict:
    obs = sorted(observations, key=lambda o: o[0])
    sold = days = 0.0
    used = censored = skipped = 0

    for (t0, q0), (t1, q1) in zip(obs, obs[1:]):
        dt = (t1 - t0).total_seconds() / 86400
        if dt <= 0:
            continue
        if deliveries is None:
            if q1 > q0:
                skipped += 1
                continue
            available = q0
        else:
            available = q0 + sum(u for d, u in deliveries if t0 < d <= t1)
            if q1 > available:
                skipped += 1
                continue
        if q1 == 0:
            censored += 1
            continue
        sold += available - q1
        days += dt
        used += 1

    demand: Optional[float] = round(sold / days, 3) if days >= MIN_OBSERVED_DAYS else None
    total = used + censored
    censored_share = censored / total if total else 0.0
    confidence = 0.0
    if demand is not None:
        confidence = round(0.9 * min(1.0, days / FULL_CONFIDENCE_DAYS) * (1 - 0.5 * censored_share), 2)

    return {
        "daily_demand": demand,
        "observed_days": round(days, 2),
        "intervals_used": used,
        "intervals_censored": censored,
        "intervals_skipped_restock": skipped,  # restocks or unexplained increases
        "demand_is_lower_bound": censored > 0,
        "confidence": confidence,  # heuristic 0-1
        "method": "pooled_depletion_rate" + ("+ledger" if deliveries is not None else ""),
    }
