import json
import math
import os
import pathlib
import sys
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

os.environ["SHELFPRINT_SERVICE_TOKEN"] = "test-token"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api import app
from app import assess_opportunity, assess_retailer, calculate_recommended_units
from demand_estimation import estimate_daily_demand
from validation import ValidationError, validate_record

ROWS = json.loads((pathlib.Path(__file__).resolve().parents[1] / "sample_data.json").read_text())
client = TestClient(app, headers={"X-Service-Token": "test-token"})


def base(**kw):
    r = dict(retailer_id="R1", product_id="P1", current_stock=10, daily_demand=2,
             target_coverage_days=5, lead_time_days=2, safety_stock=1)
    r.update(kw)
    return r


def test_sample_scenarios():
    a, b, c = (assess_opportunity(r) for r in ROWS)
    assert (a["stockout_risk"], a["recommended_units"], a["potential_missed_units"]) == ("high", 19, 8)
    assert (b["stockout_risk"], b["recommended_units"]) == ("critical", 18)
    assert c["stockout_risk"] == "unknown" and c["recommendation"]["action"] == "collect_data"
    assert b["priority_score"] > a["priority_score"] > c["priority_score"]


def test_batch_returns_every_record_ranked():  # regression: old loop returned after record 1
    r = client.post("/assess/batch", json={"retailers": ROWS}).json()
    assert r["total_assessed"] == 3 and r["total_rejected"] == 0
    assert [x["rank"] for x in r["results"]] == [1, 2, 3]
    assert r["results"][0]["retailer_id"] == "SHOP-002"


def test_batch_isolates_duplicates():
    r = client.post("/assess/batch", json={"retailers": [ROWS[0], ROWS[0]]}).json()
    assert r["total_assessed"] == 1 and r["rejected"][0]["issues"][0]["code"] == "duplicate_in_batch"


@pytest.mark.parametrize("bad", [
    dict(current_stock=-1), dict(daily_demand=-1), dict(daily_demand=float("nan")),
    dict(daily_demand=float("inf")), dict(evidence_confidence=1.5), dict(order_multiple=0),
    dict(current_stock="12"), dict(current_stock=True),
])
def test_engine_rejects_bad_input(bad):
    with pytest.raises(ValidationError):
        assess_retailer(base(**bad))


def test_api_rejects_nan_and_unknown_fields():
    assert client.post("/assess", json={**base(), "surprise": 1}).status_code == 422
    assert client.post("/assess", data=json.dumps({**base(), "daily_demand": "NaN"}),
                       headers={"content-type": "application/json"}).status_code == 422


def test_zero_stock_unknown_demand_is_critical():
    a = assess_retailer(base(current_stock=0, daily_demand=None))
    assert a["stockout_risk"] == "critical" and a["days_of_cover"] is None


def test_boundary_not_flipped_by_rounding():
    # cover = 4.004 days > lead 4 -> must be "high", not "critical"
    assert assess_retailer(base(current_stock=1001, daily_demand=250, lead_time_days=4))["stockout_risk"] == "high"


def test_float_noise_does_not_add_a_unit():
    assert calculate_recommended_units(0, 0.1 * 3, 10, 0, 0) == 3


def test_order_multiple_rounds_up():
    assert calculate_recommended_units(0, 2, 5, 0, 0, order_multiple=6) == 12


def test_warnings_surface():
    codes = {i.code for i in validate_record(base(daily_demand=None, confirmed_stockout_days=3))}
    assert {"no_demand_estimate", "stockout_without_demand"} <= codes


def test_estimator_skips_restock_and_flags_censoring():
    t0 = datetime(2026, 10, 1)
    obs = [(t0, 100), (t0 + timedelta(days=5), 50),      # sold 50 / 5d
           (t0 + timedelta(days=6), 200),                # restock -> skipped
           (t0 + timedelta(days=10), 120),               # sold 80 / 4d
           (t0 + timedelta(days=12), 0)]                 # censored
    e = estimate_daily_demand(obs)
    assert math.isclose(e["daily_demand"], 130 / 9, abs_tol=0.01)
    assert e["intervals_skipped_restock"] == 1 and e["demand_is_lower_bound"] is True


def test_estimator_returns_none_when_data_too_thin():
    t0 = datetime(2026, 10, 1)
    assert estimate_daily_demand([(t0, 10), (t0 + timedelta(days=1), 8)])["daily_demand"] is None
