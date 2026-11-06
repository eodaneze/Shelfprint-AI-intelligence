import pytest
from app import (
    calculate_days_of_cover,
    calculate_recommended_units,
    estimate_potential_missed_units,
    assess_opportunity,
    score_opportunity,
)


def test_days_of_cover():
    assert calculate_days_of_cover(12, 4) == 3.0


def test_unknown_demand_returns_none():
    assert calculate_days_of_cover(12, None) is None


def test_replenishment():
    # target stock = 4 * (5 + 2) + 3 = 31; 31 - 12 = 19
    assert calculate_recommended_units(12, 4, 5, 2, 3) == 19


def test_replenishment_order_multiple():
    assert calculate_recommended_units(12, 4, 5, 2, 3, order_multiple=6) == 24


def test_missed_units_are_rounded_up():
    assert estimate_potential_missed_units(2.2, 2) == 5


def test_missing_missed_demand_returns_none():
    assert estimate_potential_missed_units(None, 2) is None


def test_negative_stock_rejected():
    with pytest.raises(ValueError):
        calculate_days_of_cover(-1, 2)


def test_priority_score():
    assert score_opportunity(100, 80, 90, 60) == 86.0


def test_score_rejects_out_of_range():
    with pytest.raises(ValueError):
        score_opportunity(101, 80, 90, 60)


def test_assess_opportunity_returns_priority_and_breakdown():
    record = {
        "retailer_id": "SHOP-001",
        "product_id": "SKU-004",
        "current_stock": 12,
        "daily_demand": 4,
        "target_coverage_days": 5,
        "lead_time_days": 2,
        "safety_stock": 3,
        "confirmed_stockout_days": 2,
        "evidence_confidence": 0.9,
        "order_multiple": 1,
    }

    result = assess_opportunity(record)

    assert result["priority_score"] == 77.5
    assert result["priority_breakdown"] == {
        "urgency": 80,
        "recoverable_demand": 80,
        "evidence_confidence": 90,
        "delivery_efficiency": 50,
    }
