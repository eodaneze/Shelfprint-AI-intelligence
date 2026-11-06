"""Offline evaluation of shelf detections against human-labelled ground truth.

Contract asks for: precision/recall by SKU, quantity absolute error, calibration, human-correction feedback.
Sample = {"truth": {productId: qty}, "detections": [{"productId", "suggestedQuantity", "confidence"}]}.
Feed it the human-corrected counts from confirmed reports as ground truth.
"""
from collections import defaultdict

BIN_EDGES = [0.5, 0.7, 0.9]
BIN_LABELS = ["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]


def _bin(conf: float) -> int:
    return sum(conf >= e for e in BIN_EDGES)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p and round(p, 3), "recall": r and round(r, 3), "f1": f1 and round(f1, 3)}


def evaluate(samples: list) -> dict:
    per = defaultdict(lambda: [0, 0, 0])   # tp, fp, fn per product
    abs_errors, exact, within1, bins = [], 0, 0, defaultdict(lambda: [0, 0])  # bin -> [n, correct]
    for s in samples:
        truth = {p: q for p, q in s["truth"].items() if q > 0}
        seen = set()
        for d in s["detections"]:
            pid, qty, conf = d["productId"], d["suggestedQuantity"], d["confidence"]
            if pid in truth and pid not in seen:
                seen.add(pid)
                per[pid][0] += 1
                err = abs(qty - truth[pid])
                abs_errors.append(err)
                exact += err == 0
                within1 += err <= 1
                bins[_bin(conf)][0] += 1
                bins[_bin(conf)][1] += err == 0
            else:
                per[pid][1] += 1              # unknown to truth, or duplicate: counted as a false positive
                bins[_bin(conf)][0] += 1
        for pid in truth.keys() - seen:
            per[pid][2] += 1
    tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values()); fn = sum(v[2] for v in per.values())
    n = len(abs_errors)
    return {
        "overall": _prf(tp, fp, fn),
        "quantity": {"matched": n, "mae": round(sum(abs_errors) / n, 3) if n else None,
                     "exact_match_rate": round(exact / n, 3) if n else None,
                     "within_one_rate": round(within1 / n, 3) if n else None},
        "by_product": {pid: _prf(*v) for pid, v in sorted(per.items())},
        "calibration": [{"confidence_bin": BIN_LABELS[i], "n": bins[i][0],
                         "exact_correct_rate": round(bins[i][1] / bins[i][0], 3)} for i in sorted(bins)],
    }
