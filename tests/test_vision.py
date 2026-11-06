import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluate import evaluate
from vision import AnthropicProvider, StubProvider, VisionError, analyze, parse_model_json, validate_image_url

REQ = {"requestId": "r1", "organizationId": "o1", "reportId": "rep1",
       "images": [{"imageId": "img1", "url": "https://res.cloudinary.com/demo/a.jpg"}],
       "catalog": [{"productId": "p1", "sku": "PEAK-400G", "name": "Peak 400g"},
                   {"productId": "p2", "sku": "MILO-200G", "name": "Milo 200g"}]}


def reply(obj, fenced=True):
    txt = json.dumps(obj)
    return 200, {"content": [{"type": "text", "text": f"```json\n{txt}\n```" if fenced else txt}]}


def provider(post, **kw):
    return AnthropicProvider(api_key="k", post=post, sleep=lambda s: None, **kw)


def raises(code, fn):
    try:
        fn()
    except VisionError as e:
        assert e.code == code, (e.code, code)
        return e
    raise AssertionError("expected VisionError " + code)


def test_stub_is_deterministic_labelled_and_valid():
    a, b = analyze(REQ, StubProvider()), analyze(REQ, StubProvider())
    assert a["detections"] == b["detections"] and a["simulated"] is True and a["valid"] is True
    assert all(d["needsReview"] for d in a["detections"]) and a["rejectedDetections"] == []
    assert any("SIMULATED" in w for w in a["warnings"])


def test_anthropic_happy_path_filters_invented_products():
    ok = {"productId": "p1", "sku": "PEAK-400G", "suggestedQuantity": 8.0, "confidence": 0.87,
          "evidence": [{"imageId": "img1", "boundingBox": [0.1, 0.2, 0.35, 0.65]}]}
    ghost = {**ok, "productId": "ghost", "sku": "X"}
    out = analyze(REQ, provider(lambda *a: reply({"detections": [ok, ghost], "unmatchedObjects": 1, "warnings": ["blur"]})))
    assert [d["productId"] for d in out["detections"]] == ["p1"] and out["detections"][0]["suggestedQuantity"] == 8
    assert out["unmatchedObjects"] == 2 and out["simulated"] is False and "blur" in out["warnings"]
    assert out["modelVersion"].startswith("anthropic:")


def test_request_body_is_catalogue_constrained_and_has_no_sampling_params():
    seen = {}
    def post(url, headers, body, timeout):
        seen.update(body=body, headers=headers)
        return reply({"detections": []})
    analyze(REQ, provider(post))
    b = seen["body"]
    assert not {"temperature", "top_p", "top_k"} & set(b) and "ONLY products from the provided catalogue" in b["system"]
    assert any(c["type"] == "image" for c in b["messages"][0]["content"])
    assert "p2" in b["messages"][0]["content"][-1]["text"] and seen["headers"]["x-api-key"] == "k"


def test_unparseable_output_returns_empty_with_warning():
    out = analyze(REQ, provider(lambda *a: (200, {"content": [{"type": "text", "text": "sorry, cannot help"}]})))
    assert out["detections"] == [] and any("unparseable" in w for w in out["warnings"])


def test_retry_then_success_and_failure_codes():
    calls = []
    def flaky(*a):
        calls.append(1)
        return (529, {}) if len(calls) == 1 else reply({"detections": []})
    assert analyze(REQ, provider(flaky))["detections"] == [] and len(calls) == 2
    raises("provider_error", lambda: analyze(REQ, provider(lambda *a: (400, {}))))
    raises("provider_timeout", lambda: analyze(REQ, provider(lambda *a: (_ for _ in ()).throw(TimeoutError()))))
    raises("provider_error", lambda: analyze(REQ, provider(lambda *a: (503, {}))))


def test_missing_api_key():
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    raises("provider_not_configured", lambda: AnthropicProvider())


def test_image_url_rules():
    for bad in ["http://res.cloudinary.com/a.jpg", "https://localhost/a.jpg", "https://127.0.0.1/a.jpg",
                "https://user:pw@res.cloudinary.com/a.jpg", "https://db.internal/a.jpg", "ftp://x/a"]:
        raises("invalid_image_url", lambda u=bad: validate_image_url(u, "i"))
    raises("invalid_image_url", lambda: validate_image_url("https://evil.com/a.jpg", "i", ["res.cloudinary.com"]))
    validate_image_url("https://res.cloudinary.com/x.jpg", "i", ["cloudinary.com"])
    e = raises("invalid_image_url", lambda: validate_image_url("http://x.com/a?sig=SECRET", "i"))
    assert "SECRET" not in str(e)


def test_stub_disabled_in_production_unless_allowed():
    import os
    from vision import get_provider
    old = {k: os.environ.get(k) for k in ("SHELFPRINT_ENV", "SHELFPRINT_VISION_PROVIDER", "SHELFPRINT_ALLOW_STUB")}
    try:
        os.environ.update(SHELFPRINT_ENV="production", SHELFPRINT_VISION_PROVIDER="stub")
        os.environ.pop("SHELFPRINT_ALLOW_STUB", None)
        raises("provider_not_configured", get_provider)
        os.environ["SHELFPRINT_ALLOW_STUB"] = "1"
        assert get_provider().name == "stub"
        os.environ["SHELFPRINT_ENV"] = "development"
        os.environ.pop("SHELFPRINT_ALLOW_STUB")
        assert get_provider().name == "stub"
    finally:
        for k, val in old.items():
            os.environ.pop(k, None) if val is None else os.environ.__setitem__(k, val)


def test_parse_model_json():
    assert parse_model_json('noise {"a": 1} tail') == {"a": 1}
    assert parse_model_json("no json") is None and parse_model_json("{bad") is None


def test_evaluate_metrics():
    r = evaluate([
        {"truth": {"p1": 5, "p2": 3}, "detections": [{"productId": "p1", "suggestedQuantity": 5, "confidence": 0.9},
                                                       {"productId": "p2", "suggestedQuantity": 4, "confidence": 0.6}]},
        {"truth": {"p1": 2}, "detections": [{"productId": "p1", "suggestedQuantity": 2, "confidence": 0.95},
                                              {"productId": "p3", "suggestedQuantity": 1, "confidence": 0.8}]}])
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (3, 1, 0)
    assert r["overall"]["precision"] == 0.75 and r["overall"]["recall"] == 1.0
    assert r["quantity"]["mae"] == 0.333 and r["quantity"]["exact_match_rate"] == 0.667
    cal = {c["confidence_bin"]: (c["n"], c["exact_correct_rate"]) for c in r["calibration"]}
    assert cal == {">=0.9": (2, 1.0), "0.7-0.9": (1, 0.0), "0.5-0.7": (1, 0.0)}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
