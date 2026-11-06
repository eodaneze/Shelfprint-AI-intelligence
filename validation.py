"""Data validation for retailer stock observations.

Two severities:
  error   -> record cannot be scored safely (bad type, negative, NaN, missing).
  warning -> record is scorable but evidence is weak or suspicious; surfaced in the UI.
"""
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real

# Tunable plausibility thresholds (calibrate on the seeded data)
MAX_PLAUSIBLE_DAILY_DEMAND = 500      # units/day for one small shop-SKU
MAX_STOCKOUT_DAYS = 30
LONG_LEAD_TIME_DAYS = 14
OVERSTOCK_COVER_DAYS = 60
LOW_EVIDENCE = 0.4


@dataclass(frozen=True)
class Issue:
    field: str
    severity: str  # "error" | "warning"
    code: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class ValidationError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("; ".join(f"{i.field}: {i.message}" for i in self.issues))


def _is_number(v) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool) and isfinite(v)


def validate_record(record: dict) -> list:
    issues: list = []

    def err(field, code, msg):
        issues.append(Issue(field, "error", code, msg))

    def warn(field, code, msg):
        issues.append(Issue(field, "warning", code, msg))

    for f in ("retailer_id", "product_id"):
        v = record.get(f)
        if not isinstance(v, str) or not v.strip():
            err(f, "missing_id", f"{f} is required.")

    for f in ("current_stock", "target_coverage_days", "lead_time_days", "safety_stock"):
        v = record.get(f)
        if v is None:
            err(f, "missing", f"{f} is required.")
        elif not _is_number(v):
            err(f, "not_numeric", f"{f} must be a finite number.")
        elif v < 0:
            err(f, "negative", f"{f} cannot be negative.")
        elif v != int(v):
            err(f, "not_integer", f"{f} must be a whole number.")

    for f in ("daily_demand", "confirmed_stockout_days", "observed_unmet_units"):
        v = record.get(f)
        if v is None:
            continue
        if not _is_number(v):
            err(f, "not_numeric", f"{f} must be a finite number or null.")
        elif v < 0:
            err(f, "negative", f"{f} cannot be negative.")

    conf = record.get("evidence_confidence", 0.5)
    if not _is_number(conf) or not 0 <= conf <= 1:
        err("evidence_confidence", "out_of_range", "evidence_confidence must be between 0 and 1.")

    om = record.get("order_multiple", 1)
    if not _is_number(om) or om <= 0 or om != int(om):
        err("order_multiple", "invalid", "order_multiple must be a positive whole number.")

    if any(i.severity == "error" for i in issues):
        return issues  # types are unsafe; skip business rules

    stock = record["current_stock"]
    demand = record.get("daily_demand")
    stockout_days = record.get("confirmed_stockout_days")

    if demand is None or demand == 0:
        warn("daily_demand", "no_demand_estimate",
             "No usable demand estimate: cover, forecast and missed units are unavailable.")
    elif demand > MAX_PLAUSIBLE_DAILY_DEMAND:
        warn("daily_demand", "demand_outlier", f"daily_demand above {MAX_PLAUSIBLE_DAILY_DEMAND}/day; check units.")
    elif stock > OVERSTOCK_COVER_DAYS * demand:
        warn("current_stock", "possible_overstock", f"Stock exceeds {OVERSTOCK_COVER_DAYS} days of demand.")

    if stockout_days is not None:
        if stockout_days > MAX_STOCKOUT_DAYS:
            warn("confirmed_stockout_days", "stockout_days_long", f"More than {MAX_STOCKOUT_DAYS} days; verify.")
        if stockout_days > 0 and demand is None:
            warn("confirmed_stockout_days", "stockout_without_demand", "Missed units cannot be estimated without demand.")
        if stockout_days > 0 and stock > 0:
            warn("confirmed_stockout_days", "stockout_stock_conflict",
                 "Stock is positive although a stockout was reported; confirm the stockout has ended.")

    if record["lead_time_days"] > LONG_LEAD_TIME_DAYS:
        warn("lead_time_days", "long_lead_time", f"Lead time above {LONG_LEAD_TIME_DAYS} days; verify.")
    if conf < LOW_EVIDENCE:
        warn("evidence_confidence", "low_evidence", "Low evidence confidence; require manual confirmation.")

    return issues
