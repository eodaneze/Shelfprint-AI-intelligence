import pathlib
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from demand_estimation import estimate_daily_demand
from opportunities import compute_opportunities
from shelf_validation import validate_analysis_response

AS_OF = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def day(n):
    return (AS_OF - timedelta(days=n)).isoformat()


def item(**kw):
    base = dict(retailerId="R1", productId="P1", onHandUnits=10, observedAt=day(0),
                countHistory=[{"observedAt": day(10), "quantity": 80}, {"observedAt": day(5), "quantity": 45}],
                deliveries=None, signals=[], confirmedReportIds=["rep1"])
    base.update(kw)
    return base


def test_depletion_demand_and_contract_formula():
    # 80->45 over 5d (7/day), 45->10 over 5d (7/day) => 7/day; lead=0, safety=0 => ceil(7*7) - 10 = 39
    o = compute_opportunities([item()], AS_OF)["items"][0]
    assert abs(o["estimatedDailyDemandUnits"] - 7.0) < 1e-6
    assert o["recommendedUnits"] == 39 and o["daysOfCover"] == 1.43
    assert o["stockoutRisk"] == "high" and o["demandSource"] == "depletion"
    assert o["evidence"]["confirmedReportIds"] == ["rep1"] and o["assumptions"]


def test_ledger_recovers_restock_interval():
    hist = [{"observedAt": day(10), "quantity": 20}, {"observedAt": day(5), "quantity": 60}]
    no_ledger = estimate_daily_demand([(datetime.fromisoformat(h["observedAt"]), h["quantity"]) for h in hist])
    assert no_ledger["daily_demand"] is None  # restock interval skipped, too little data
    pts = [(AS_OF - timedelta(days=d), q) for d, q in [(10, 20), (5, 60), (0, 30)]]
    led = estimate_daily_demand(pts, [(AS_OF - timedelta(days=7), 50)])  # 20+50-60=10 sold /5d; 60-30=30 /5d
    assert abs(led["daily_demand"] - 4.0) < 1e-6 and led["method"].endswith("+ledger")


def test_requests_add_unmet_demand_and_observed_vs_inferred():
    sigs = [{"id": "s1", "type": "customer_request", "quantity": 6, "observedAt": day(2)},
            {"id": "s2", "type": "stockout_observation", "quantity": 4, "observedAt": day(1)}]
    o = compute_opportunities([item(signals=sigs)], AS_OF)["items"][0]
    assert o["demandSource"] == "depletion+requests"
    assert o["missedUnitsBreakdown"]["observedUnmetUnits"] == 10
    assert o["evidence"]["signalIds"] == ["s1", "s2"]
    assert o["estimatedDailyDemandUnits"] > 7.0


def test_requests_only_and_needs_data():
    sigs = [{"id": "s1", "type": "customer_request", "quantity": 3, "observedAt": day(1)}]
    out = compute_opportunities([item(countHistory=[], signals=sigs),
                                 item(retailerId="R2", countHistory=[], onHandUnits=0)], AS_OF)
    assert out["items"][0]["demandSource"] == "requests_only" and out["items"][0]["confidence"] <= 0.4
    assert out["needsData"][0]["retailerId"] == "R2" and out["needsData"][0]["stockedOut"] is True


def test_stale_count_lowers_confidence_and_warns():
    fresh = compute_opportunities([item()], AS_OF)["items"][0]
    stale = compute_opportunities([item(observedAt=day(9), countHistory=[
        {"observedAt": day(19), "quantity": 80}, {"observedAt": day(14), "quantity": 45}])], AS_OF)["items"][0]
    assert stale["confidence"] < fresh["confidence"]
    assert any(i["code"] == "stale_stock_count" for i in stale["dataQuality"])


def test_ranking_duplicates_and_future_counts():
    out = compute_opportunities([item(), item(retailerId="R2", onHandUnits=500), item(),
                                 item(retailerId="R3", observedAt=(AS_OF + timedelta(days=1)).isoformat())], AS_OF)
    assert [o["retailerId"] for o in out["items"]] == ["R1", "R2"] and out["items"][0]["rank"] == 1
    assert {s["reason"] for s in out["skipped"]} == {"duplicate_in_request", "count_observed_after_asOf"}


REQ = {"requestId": "r1", "images": [{"imageId": "img1", "url": "x"}],
       "catalog": [{"productId": "p1", "sku": "PEAK-400G"}, {"productId": "p2", "sku": "MILO-200G"}]}


def det(**kw):
    d = {"productId": "p1", "sku": "PEAK-400G", "suggestedQuantity": 8, "confidence": 0.87, "needsReview": False,
         "evidence": [{"imageId": "img1", "boundingBox": [0.1, 0.2, 0.35, 0.65]}]}
    d.update(kw)
    return d


def test_shelf_validation_rejects_invented_and_malformed():
    resp = {"requestId": "r1", "modelVersion": "shelf-v0.1", "unmatchedObjects": 2, "detections": [
        det(), det(productId="ghost", sku="X"), det(productId="p2", sku="WRONG"),
        det(productId="p2", sku="MILO-200G", suggestedQuantity=-1),
        det(productId="p2", sku="MILO-200G", evidence=[{"imageId": "img1", "boundingBox": [0.5, 0.2, 0.3, 0.9]}])]}
    out = validate_analysis_response(REQ, resp)
    assert [d["productId"] for d in out["detections"]] == ["p1"]
    assert out["detections"][0]["needsReview"] is True       # advisory, forced
    assert out["unmatchedObjects"] == 3                       # 2 reported + 1 invented product
    reasons = [r["reasons"] for r in out["rejectedDetections"]]
    assert ["unknown_product"] in reasons and ["sku_mismatch"] in reasons and ["invalid_quantity"] in reasons
    assert ["invalid_bounding_box"] in reasons


def test_shelf_validation_duplicates_and_envelope():
    resp = {"requestId": "other", "detections": [det(suggestedQuantity=8, confidence=0.9),
                                                  det(suggestedQuantity=11, confidence=0.4)]}
    out = validate_analysis_response(REQ, resp)
    assert out["valid"] is False and len(out["detections"]) == 1
    assert out["detections"][0]["suggestedQuantity"] == 8 and out["detections"][0]["alternativeQuantities"] == [11]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
