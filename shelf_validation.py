"""Deterministic validation of the vision model's /v1/shelf/analyze output against the request.

Rules from the contract: catalogue-constrained (never invent SKU/product IDs), normalized bounding
boxes, AI output is advisory (needsReview is always true), unknown products stay unmatched.
"""
from math import isfinite

from validation import Issue

MAX_QUANTITY = 500
LOW_CONFIDENCE = 0.5


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)


def _bbox_ok(b) -> bool:
    if not isinstance(b, (list, tuple)) or len(b) != 4 or not all(_num(x) and 0 <= x <= 1 for x in b):
        return False
    return b[0] < b[2] and b[1] < b[3]


def _check_detection(d, catalog, image_ids) -> list:
    if not isinstance(d, dict):
        return ["invalid_shape"]
    reasons = []
    pid = d.get("productId")
    if pid not in catalog:
        reasons.append("unknown_product")
    elif d.get("sku") != catalog[pid].get("sku"):
        reasons.append("sku_mismatch")
    q = d.get("suggestedQuantity")
    if not isinstance(q, int) or isinstance(q, bool) or not 0 <= q <= MAX_QUANTITY:
        reasons.append("invalid_quantity")
    c = d.get("confidence")
    if not _num(c) or not 0 <= c <= 1:
        reasons.append("invalid_confidence")
    ev = d.get("evidence")
    if not isinstance(ev, list) or not ev:
        reasons.append("missing_evidence")
    else:
        for e in ev:
            if not isinstance(e, dict) or e.get("imageId") not in image_ids:
                reasons.append("unknown_image")
            elif not _bbox_ok(e.get("boundingBox")):
                reasons.append("invalid_bounding_box")
    return sorted(set(reasons))


def validate_analysis_response(request: dict, response: dict) -> dict:
    catalog = {c["productId"]: c for c in request.get("catalog") or []}
    image_ids = {i["imageId"] for i in request.get("images") or []}
    issues, warnings, rejected, accepted = [], [], [], []

    mv = response.get("modelVersion")
    if not isinstance(mv, str) or not mv.strip():
        issues.append(Issue("modelVersion", "error", "missing_model_version", "modelVersion is required."))
    if response.get("requestId") != request.get("requestId"):
        issues.append(Issue("requestId", "error", "request_id_mismatch", "Response requestId does not match the request."))
    warnings += [w for w in response.get("warnings") or [] if isinstance(w, str)]

    for idx, d in enumerate(response.get("detections") or []):
        reasons = _check_detection(d, catalog, image_ids)
        if reasons:
            rejected.append({"index": idx, "productId": d.get("productId") if isinstance(d, dict) else None,
                             "reasons": reasons})
        else:
            accepted.append({**d, "needsReview": True, "lowConfidence": d["confidence"] < LOW_CONFIDENCE})

    # One item per product (report review rejects duplicate SKUs). Never sum: images may overlap.
    by_product = {}
    for d in accepted:
        by_product.setdefault(d["productId"], []).append(d)
    final = []
    for pid, group in by_product.items():
        group.sort(key=lambda d: -d["confidence"])
        keep = dict(group[0])
        if len(group) > 1:
            keep["alternativeQuantities"] = sorted({g["suggestedQuantity"] for g in group[1:]})
            warnings.append(f"Multiple detections for {pid}; kept the highest-confidence one, review required.")
        final.append(keep)
    final.sort(key=lambda d: d["productId"])

    unmatched = response.get("unmatchedObjects")
    unmatched = unmatched if isinstance(unmatched, int) and not isinstance(unmatched, bool) and unmatched >= 0 else 0
    unmatched += sum(1 for r in rejected if "unknown_product" in r["reasons"])

    return {
        "requestId": request.get("requestId"),
        "modelVersion": mv if isinstance(mv, str) else None,
        "valid": not any(i.severity == "error" for i in issues),
        "detections": final,
        "rejectedDetections": rejected,
        "unmatchedObjects": unmatched,
        "warnings": warnings,
        "issues": [i.to_dict() for i in issues],
        "needsReview": True,
    }
