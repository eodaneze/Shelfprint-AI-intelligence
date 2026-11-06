
from dotenv import load_dotenv

load_dotenv()

import hmac
import os
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import DEFAULT_DELIVERY_EFFICIENCY, MODEL_VERSION, assess_opportunity
from opportunities import MODEL_VERSION as OPP_VERSION, compute_opportunities
from shelf_validation import validate_analysis_response
from validation import ValidationError, validate_record
from vision import DEFAULT_MODEL, VisionError, analyze, provider_name

app = FastAPI(title="SHELFPRINT AI Demand Intelligence API", version="0.3.0")
MAX_BATCH = 500


def require_service_token(x_service_token: Optional[str] = Header(default=None)):
    """Internal Node service token. Set SHELFPRINT_SERVICE_TOKEN in the environment."""
    expected = os.getenv("SHELFPRINT_SERVICE_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Service token is not configured.")
    if not x_service_token or not hmac.compare_digest(x_service_token, expected):
        raise HTTPException(status_code=401, detail="Invalid service token.")


protected = APIRouter(dependencies=[Depends(require_service_token)])


# ---------- models ----------
class RetailerStockInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retailer_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    current_stock: int = Field(ge=0, le=1_000_000)
    daily_demand: Optional[float] = Field(default=None, ge=0, le=100_000, allow_inf_nan=False)
    target_coverage_days: int = Field(ge=0, le=365)
    lead_time_days: int = Field(ge=0, le=365)
    safety_stock: int = Field(ge=0, le=1_000_000)
    confirmed_stockout_days: Optional[float] = Field(default=None, ge=0, le=365, allow_inf_nan=False)
    evidence_confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    order_multiple: int = Field(default=1, gt=0, le=10_000)


class BatchAssessmentInput(BaseModel):
    retailers: list[RetailerStockInput] = Field(min_length=1, max_length=MAX_BATCH)


class CountPoint(BaseModel):
    observedAt: datetime
    quantity: int = Field(ge=0, le=1_000_000)


class DeliveryPoint(BaseModel):
    deliveredAt: datetime
    units: int = Field(ge=1, le=1_000_000)


class SignalIn(BaseModel):
    id: Optional[str] = None
    type: Literal["customer_request", "stockout_observation"]
    quantity: int = Field(ge=1, le=100_000)
    observedAt: datetime


class OpportunityItemIn(BaseModel):
    retailerId: str = Field(min_length=1)
    productId: str = Field(min_length=1)
    onHandUnits: int = Field(ge=0, le=1_000_000)   # confirmed count; never send unobserved stock
    observedAt: datetime
    unitsPerCarton: Optional[int] = Field(default=None, ge=1, le=10_000)
    countHistory: list[CountPoint] = Field(default_factory=list, max_length=500)
    deliveries: Optional[list[DeliveryPoint]] = Field(default=None, max_length=500)
    signals: list[SignalIn] = Field(default_factory=list, max_length=1000)
    confirmedReportIds: list[str] = Field(default_factory=list, max_length=500)


class OpportunityRequest(BaseModel):
    asOf: Optional[datetime] = None
    targetCoverDays: int = Field(default=7, ge=0, le=365)
    leadTimeDays: int = Field(default=0, ge=0, le=365)
    safetyStock: int = Field(default=0, ge=0, le=1_000_000)
    windowDays: int = Field(default=14, ge=3, le=90)
    items: list[OpportunityItemIn] = Field(min_length=1, max_length=1000)


class VisionImage(BaseModel):
    imageId: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=8, max_length=2048)


class CatalogItem(BaseModel):
    productId: str = Field(min_length=1, max_length=64)
    sku: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    brand: Optional[str] = Field(default=None, max_length=100)
    referenceImageUrls: list[str] = Field(default_factory=list, max_length=10)


class ShelfAnalyzeRequest(BaseModel):
    requestId: str = Field(min_length=1, max_length=64)
    organizationId: str = Field(min_length=1, max_length=64)
    reportId: str = Field(min_length=1, max_length=64)
    images: list[VisionImage] = Field(min_length=1, max_length=5)
    catalog: list[CatalogItem] = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def _unique_ids(self):
        if len({i.imageId for i in self.images}) != len(self.images):
            raise ValueError("imageId values must be unique")
        if len({c.productId for c in self.catalog}) != len(self.catalog):
            raise ValueError("catalog productId values must be unique")
        return self


class ShelfValidationRequest(BaseModel):
    request: dict
    response: dict


def _issues(exc: ValidationError) -> list:
    return [i.to_dict() for i in exc.issues]


# ---------- public (health only) ----------
@app.get("/health")
@app.get("/v1/health")
def health():
    return {"status": "ok", "service": "shelfprint-demand-intelligence",
            "modelVersion": f"{OPP_VERSION}/{MODEL_VERSION}", "visionProvider": provider_name(),
            "visionModel": os.getenv("SHELFPRINT_VISION_MODEL", DEFAULT_MODEL) if provider_name() == "anthropic" else None,
            "environment": os.getenv("SHELFPRINT_ENV", "development")}


# ---------- contract endpoints for the Node backend ----------
@protected.post("/v1/demand/opportunities")
def demand_opportunities(data: OpportunityRequest):
    payload = data.model_dump()
    return compute_opportunities(
        payload["items"], as_of=payload["asOf"], target_cover_days=payload["targetCoverDays"],
        lead_time_days=payload["leadTimeDays"], safety_stock=payload["safetyStock"],
        window_days=payload["windowDays"],
    )


@protected.post("/v1/shelf/analyze")
def shelf_analyze(data: ShelfAnalyzeRequest):
    try:
        return analyze(data.model_dump())
    except VisionError as exc:
        raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)})


@protected.post("/v1/shelf/validate-detections")
def validate_detections(data: ShelfValidationRequest):
    return validate_analysis_response(data.request, data.response)


# ---------- single-record engine endpoints (dev / debugging) ----------
@protected.post("/validate")
def validate(data: RetailerStockInput):
    issues = validate_record(data.model_dump())
    return {"valid": not any(i.severity == "error" for i in issues),
            "issues": [i.to_dict() for i in issues]}


@protected.post("/assess")
def assess_stock(data: RetailerStockInput):
    try:
        return assess_opportunity(data.model_dump(), delivery_efficiency=DEFAULT_DELIVERY_EFFICIENCY)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_issues(exc))


@protected.post("/assess/batch")
def assess_batch(data: BatchAssessmentInput):
    results, rejected, seen = [], [], set()
    for index, retailer in enumerate(data.retailers):
        record = retailer.model_dump()
        key = (record["retailer_id"], record["product_id"])
        if key in seen:
            rejected.append({"index": index, "retailer_id": key[0], "product_id": key[1],
                             "issues": [{"field": "retailer_id/product_id", "severity": "error",
                                         "code": "duplicate_in_batch",
                                         "message": "Duplicate retailer/product pair in this batch."}]})
            continue
        seen.add(key)
        try:
            results.append(assess_opportunity(record, delivery_efficiency=DEFAULT_DELIVERY_EFFICIENCY))
        except ValidationError as exc:
            rejected.append({"index": index, "retailer_id": key[0], "product_id": key[1], "issues": _issues(exc)})

    results.sort(key=lambda r: (-r["priority_score"], r["retailer_id"], r["product_id"]))
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return {"total_received": len(data.retailers), "total_assessed": len(results),
            "total_rejected": len(rejected), "ranking_method": "weighted_priority_score",
            "results": results, "rejected": rejected}


app.include_router(protected)
