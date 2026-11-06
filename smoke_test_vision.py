"""Send ONE real shelf photo through the real vision model and print the result.

Usage (project root, key already set in this terminal):
    python smoke_test_vision.py                      # step 1 only: verify the key and the model ID
    python smoke_test_vision.py https://res.cloudinary.com/<cloud>/image/upload/<shelf-photo>.jpg   # steps 1 and 2
Optional: edit CATALOG below to match products that are actually in the photo.
The key is never printed.
"""
import json
import os
import sys

from vision import AnthropicProvider, VisionError, analyze

CATALOG = [  # replace with products visible in YOUR photo
    {"productId": "p1", "sku": "TOM-70G", "name": "Tomato Paste 70g Sachet", "brand": "SimBrand"},
    {"productId": "p2", "sku": "SEAS-400G", "name": "Seasoning Cubes 400g", "brand": "SimBrand"},
    {"productId": "p3", "sku": "NOOD-70G", "name": "Instant Noodles 70g", "brand": "SimBrand"},
]

if len(sys.argv) > 2:
    sys.exit(__doc__)
if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set in this terminal.")

# Step 1: is the key valid and does the configured model ID exist for this account?
import httpx
from vision import DEFAULT_MODEL

model = os.getenv("SHELFPRINT_VISION_MODEL", DEFAULT_MODEL)
r = httpx.get("https://api.anthropic.com/v1/models", params={"limit": 1000}, timeout=20,
              headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
if r.status_code == 401:
    sys.exit("Key rejected (401). Check that it was copied completely and is active.")
if r.status_code != 200:
    sys.exit(f"Could not list models (HTTP {r.status_code}).")
ids = [m["id"] for m in r.json().get("data", [])]
if model not in ids:
    sys.exit(f"Model ID '{model}' is NOT available to this key. Available: {', '.join(ids[:15])}")
print(f"OK: key works and model ID '{model}' is available.")
if len(sys.argv) == 1:
    sys.exit(0)

request = {"requestId": "smoke-1", "organizationId": "o", "reportId": "r",
           "images": [{"imageId": "img1", "url": sys.argv[1]}], "catalog": CATALOG}
try:
    result = analyze(request, AnthropicProvider())
except VisionError as exc:
    sys.exit(f"FAILED [{exc.code}] {exc}")

print(json.dumps(result, indent=2))
print(f"\nmodel={result['modelVersion']}  detections={len(result['detections'])}  "
      f"rejected={len(result['rejectedDetections'])}  ms={result['processingMs']}")
print("Compare the counts with the photo by eye. Do not trust them blindly.")
