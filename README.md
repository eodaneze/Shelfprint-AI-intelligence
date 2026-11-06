# SHELFPRINT AI: Demand Intelligence Service

Python/FastAPI service that turns confirmed retail stock evidence into ranked, explainable
replenishment opportunities, and validates AI shelf-detection output before it reaches a human reviewer.

> **Status: hackathon MVP.** All scores, weights and thresholds are transparent baselines to be calibrated
> on seeded or pilot data. Outputs are **estimates, not confirmed sales**.

## Where this fits

```
Mobile app -> Node/Express backend (auth, uploads, reports, inventory ledger, MongoDB)
                    |  internal HTTP + X-Service-Token
                    v
              This service (stateless calculator)
                - POST /v1/demand/opportunities
                - POST /v1/shelf/analyze (photo -> detections)
                - POST /v1/shelf/validate-detections
```

- **Node owns** authentication, tenancy, stock reports, inventory snapshots and movements.
- **This service owns** demand estimation, risk, recommended units, priority scoring, evidence/assumptions,
  and validation of AI detections.
- **This service never writes inventory** and holds no database. Every request carries its own evidence.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Core rules for one shop/product: days of cover, risk band, recommended units, missed units, priority score |
| `validation.py` | Input validation. Errors block scoring; warnings are surfaced but still scored |
| `demand_estimation.py` | Daily demand from confirmed stock counts (depletion rate, optional delivery ledger) |
| `opportunities.py` | Builds contract-shaped `DemandOpportunity` objects from Node's evidence |
| `vision.py` | Shelf photo analysis: providers (stub, anthropic), URL safety, retries, orchestration |
| `shelf_validation.py` | Validates vision-model output against the catalogue |
| `evaluate.py` | Offline metrics: precision/recall by SKU, quantity error, confidence calibration |
| `api.py` | FastAPI endpoints and service-token auth |
| `sample_data.json` | Three sample shop/product records for `app.py` |
| `tests/` | `test_engine.py` (engine + API), `test_contract.py` (opportunities + shelf validation), `test_vision.py` (vision + evaluation) |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py                      # prints ranked sample output (no server needed)

export SHELFPRINT_SERVICE_TOKEN=demo-token   # Windows PowerShell: $env:SHELFPRINT_SERVICE_TOKEN="demo-token"
uvicorn api:app --reload           # Swagger UI: http://127.0.0.1:8000/docs
```

Run tests: `pytest` (or `python tests/test_contract.py` and `python tests/test_vision.py` without pytest).

## Authentication

All routes except `GET /health` and `GET /v1/health` require the header `X-Service-Token`, which must equal the
`SHELFPRINT_SERVICE_TOKEN` environment variable. Missing token config returns 503; a wrong token returns 401.
Deploy this service on a private network only.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/v1/health` | Readiness and model version (public) |
| POST | `/v1/demand/opportunities` | Ranked `DemandOpportunity[]` from confirmed evidence |
| POST | `/v1/shelf/analyze` | Analyze shelf photos against the catalogue (contract section 07) |
| POST | `/v1/shelf/validate-detections` | Validate vision output against request and catalogue |
| POST | `/validate` | Validate one record (dev/debug) |
| POST | `/assess` | Score one record (dev/debug) |
| POST | `/assess/batch` | Score and rank many records (dev/debug) |

The `/assess*` and `/validate` routes are for debugging. Remove them or keep them token-protected before deploying.

### `POST /v1/demand/opportunities`

```json
{
  "asOf": "2026-11-01T12:00:00Z",
  "targetCoverDays": 7,
  "leadTimeDays": 0,
  "safetyStock": 0,
  "windowDays": 14,
  "items": [{
    "retailerId": "R1",
    "productId": "P1",
    "onHandUnits": 10,
    "observedAt": "2026-11-01T09:00:00Z",
    "unitsPerCarton": 12,
    "countHistory": [
      {"observedAt": "2026-10-22T09:00:00Z", "quantity": 80},
      {"observedAt": "2026-10-27T09:00:00Z", "quantity": 45}
    ],
    "deliveries": [],
    "signals": [{"id": "s1", "type": "customer_request", "quantity": 4, "observedAt": "2026-10-30T10:00:00Z"}],
    "confirmedReportIds": ["rep1"]
  }]
}
```

Response: `{asOf, modelVersion, items, needsData, skipped}`.

- `items`: ranked opportunities (`rank`, `score`, `stockoutRisk`, `daysOfCover`, `recommendedUnits`,
  `estimatedMissedUnits`, `missedUnitsBreakdown`, `confidence`, `evidence`, `assumptions`, `dataQuality`, ...).
- `needsData`: shops with no usable demand evidence (for example verified zero stock but no history). Not ranked.
- `skipped`: duplicates, counts dated after `asOf`, or invalid input, each with a reason.

**Input rules:** send only *confirmed* counts. Never send unobserved stock as 0. Quantities are individual saleable units.
`deliveries: null` means the ledger is unknown, so any stock increase is treated as a restock. `deliveries: []` means the
ledger is known and empty, so an unexplained increase is skipped.

### `POST /v1/shelf/analyze`

Body per the contract: `{requestId, organizationId, reportId, images:[{imageId,url}], catalog:[{productId,sku,name,brand?,referenceImageUrls?}]}`
(1-5 images, up to 300 catalogue items, unique IDs). Response: `{requestId, modelVersion, detections, rejectedDetections, unmatchedObjects,
warnings, issues, needsReview, provider, simulated, processingMs}`.

- **Providers** (`SHELFPRINT_VISION_PROVIDER`): `stub` (default) returns deterministic **fake** detections and `simulated: true`, for integration
  testing only. `anthropic` sends the images and catalogue to the Anthropic Messages API (needs `ANTHROPIC_API_KEY`; model via
  `SHELFPRINT_VISION_MODEL`).
- **Safety:** image URLs must be https, with no credentials, localhost or IP literals, and optionally match `SHELFPRINT_IMAGE_HOSTS`
  (comma-separated, for example `res.cloudinary.com`). Node still owns upload validation of actual file bytes and size.
- **Catalogue-constrained:** the prompt allows only catalogue products and treats text inside images as data. Every output also passes through
  `shelf_validation.py`, so invented products are dropped and counted as unmatched.
- **Failures:** provider unreachable returns 502, timeout returns 504, missing configuration returns 503. In those cases Node should keep the draft for manual
  review. Unparseable model output returns 200 with no detections and a warning.
- Bounding boxes from a general vision model are approximate. Model confidence is uncalibrated; measure it with `evaluate.py`.

### `POST /v1/shelf/validate-detections`

Body: `{"request": <the /v1/shelf/analyze request>, "response": <the vision model response>}`.

Returns the accepted `detections` (always `needsReview: true`), `rejectedDetections` with reasons, `unmatchedObjects`,
`warnings`, `issues` and `valid`. Detections are rejected for:

- unknown product ID
- SKU mismatch
- invalid quantity or confidence
- missing evidence
- unknown image
- invalid bounding box (must be normalized `[xMin, yMin, xMax, yMax]` with min < max)

Duplicate detections for one product are reduced to the highest-confidence one, with the other quantities listed in
`alternativeQuantities`; they are never summed because images may overlap.

## How the intelligence works

**Daily demand** = served demand + unmet demand

- *Served:* units that left the shelf between confirmed counts divided by days elapsed, pooled across intervals.
  - Restock intervals are skipped, or recovered from the delivery ledger as `before + delivered - after`.
  - Intervals ending at zero are censored: excluded, and the estimate is flagged as a lower bound.
  - Fewer than 3 observed days returns no estimate.
- *Unmet:* units from `customer_request` and `stockout_observation` signals divided by observation days
  (denominator floored at 3 days as the sparse-data fallback).

**Days of cover** = on-hand / daily demand (null if demand is unknown or zero).

**Risk bands** (lead time `L`): critical if cover <= L, high if <= L+2, medium if <= L+5, otherwise low. Zero confirmed stock is
always critical. The contract exposes `critical` as `high`; the finer value is in `riskDetail`.

**Recommended units** = `max(0, ceil(demand x (targetCoverDays + leadTime) + safetyStock - onHand))`. With lead time and safety stock at 0
(the defaults) this equals the contract formula.

**Estimated missed units** = observed unmet request units + inferred lead-time shortfall (`demand x max(0, L - cover)`).
The breakdown is returned separately so the UI can label *observed* and *inferred* differently.

**Priority score** (0-100) = 0.35 x urgency + 0.30 x recoverable demand + 0.20 x evidence confidence + 0.15 x delivery efficiency.
Urgency comes from the risk band (100/80/50/20/0). Recoverable demand = `min(100, missed units / 30 x 100)`.
Delivery efficiency is a **fixed placeholder of 50** until the route module supplies it.

**Confidence** is a heuristic number from 0 to 1 (HIGH >= 0.7, MEDIUM >= 0.4, otherwise LOW). It grows with observation days, drops with censored
intervals, and is multiplied by 0.8 if the latest count is older than 7 days. It is not a calibrated probability.

## Validation

- **Errors** block scoring: missing fields, negatives, NaN/inf, wrong types, non-integer stock, invalid `order_multiple`.
- **Warnings** still score the record and appear under `dataQuality`: no demand estimate, demand outlier, possible overstock,
  long stockout, stockout reported without demand, positive stock alongside a stockout, long lead time, low evidence, stale count.
- The API models also use `extra="forbid"`, range bounds and `allow_inf_nan=False`.

## Assumptions and limitations

- Stock unit is individual saleable units; cartons are only used for the optional `recommendedCartons`.
- A confirmed count is an absolute observation, not a sale. Depletion cannot separate sales from shrinkage or corrections.
- Missed units are estimates, never verified lost sales. Shelf-price value is not manufacturer revenue.
- Placeholders needing calibration: 14-day window, 3-day floor, 30-unit demand cap, confidence formula, staleness threshold.
- Not yet implemented: missed revenue in NGN (needs `unitPriceNGN` and a stated valuation basis), real delivery efficiency
  from routing, projected on-hand for stale counts, and a labelled evaluation set for the vision model (the `evaluate.py` harness is ready).
- FastAPI endpoint tests were written but could not be executed in the authoring environment; run `pytest` locally.

## Open decisions for the team

1. Add `unknown` to the `stockoutRisk` enum, or keep the current handling (`critical` -> `high`, no-data -> `needsData`).
2. Lead time and safety stock: per product or global? They are request-level parameters today.
3. Observation window and sparse-data fallback for demand estimation.
4. Who owns the shared end-to-end fixture dataset for the demo.

## Versions

`baseline-v1` (opportunity logic) / `rules-v0.2` (engine). Both are returned as `modelVersion` on every opportunity.
