"""HTTP API and research console (FastAPI).

Roles: ``viewer`` (model card and cohort summary), ``analyst`` (also analyse mutation files),
``admin`` (also reload the model and read the audit trail). Keys come from the environment and are
kept only as SHA-256 digests.

Uploaded files are processed in memory and never written to disk. The audit trail records who
sent what (role, SHA-256 of the body, sample count, outcome), never the content.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

from . import __version__
from .features import profiles
from .maf import Filters, MafError, read_maf
from .model import RESEARCH_USE, Model

log = logging.getLogger("oncoai")
ROLES = ("viewer", "analyst", "admin")
MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_ROWS = 100_000
MAX_SAMPLES = 25
TEXT_TYPES = ("text/plain", "text/tab-separated-values")
CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


class KeyStore:
    def __init__(self, spec: str) -> None:
        """``spec`` is 'role:key,role:key'. Keys shorter than 24 characters are refused."""
        self._keys: list[tuple[bytes, str]] = []
        for part in filter(None, (p.strip() for p in spec.split(","))):
            role, sep, key = part.partition(":")
            if not sep or role not in ROLES:
                raise ValueError(f"invalid key entry for role '{role}': use role:key with role in {ROLES}")
            if len(key) < 24:
                raise ValueError(f"key for role '{role}' is too short (minimum 24 characters)")
            self._keys.append((hashlib.sha256(key.encode()).digest(), role))
        if not self._keys:
            raise ValueError("no API keys configured")

    def role(self, key: str | None) -> str | None:
        if not key:
            return None
        digest = hashlib.sha256(key.encode()).digest()
        found = None
        for d, role in self._keys:
            if hmac.compare_digest(d, digest):
                found = role
        return found


class State:
    def __init__(self, model: Model, model_path: Path | None, example: str | None) -> None:
        self.model = model
        self.model_path = model_path
        self.example = example
        self.audit: deque[dict] = deque(maxlen=500)
        self.counters: dict[str, int] = {}
        self.analysed_samples = 0
        self.abstained = 0


def model_card(model: Model) -> dict:
    meta = model.meta
    return {
        "notice": RESEARCH_USE,
        "classes": model.classes,
        "abstain_below": model.threshold,
        "features": {"total": len(model.space.names), "genes": len(model.space.genes), "protein_changes": len(model.space.changes)},
        "cross_validation": meta.get("cv", {}),
        "per_class": meta.get("cv_per_class", {}),
        "abstention": meta.get("abstention", {}),
        "training_labels": meta.get("training_labels", {}),
        "data_source": meta.get("data_source", {}),
        "version": meta.get("oncoai_version"),
        "synthetic": bool(meta.get("synthetic", False)),
    }


def create_app(model: Model, keys: KeyStore, model_path: Path | None = None, example: str | None = None) -> FastAPI:
    app = FastAPI(
        title="Precision Oncology AI Platform",
        version=__version__,
        description=f"Mutation profiles, cohort context and tumour-type inference. {RESEARCH_USE}",
    )
    state = State(model, model_path, example)
    app.state.oncoai = state

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = (request.headers.get("x-request-id") or uuid.uuid4().hex[:16])[:64]
        request.state.rid = rid
        start = time.perf_counter()
        response = await call_next(request)
        ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["x-request-id"] = rid
        response.headers["x-content-type-options"] = "nosniff"
        response.headers["cache-control"] = "no-store"
        response.headers["referrer-policy"] = "no-referrer"
        key = f"{request.method} {response.status_code}"
        state.counters[key] = state.counters.get(key, 0) + 1
        log.info(json.dumps({"rid": rid, "method": request.method, "path": request.url.path, "status": response.status_code, "ms": ms}))
        return response

    def require(minimum: str):
        level = ROLES.index(minimum)

        def dep(x_api_key: str | None = Header(default=None)) -> str:
            role = keys.role(x_api_key)
            if role is None:
                raise HTTPException(401, "missing or invalid x-api-key")
            if ROLES.index(role) < level:
                raise HTTPException(403, f"role '{role}' cannot do this; needs '{minimum}'")
            return role

        return dep

    def audit(request: Request, role: str, action: str, outcome: str, **extra) -> None:
        entry = {
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
            "rid": request.state.rid,
            "role": role,
            "action": action,
            "outcome": outcome,
        }
        entry.update(extra)
        state.audit.appendleft(entry)
        log.info(json.dumps({"audit": entry}))

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        ready = state.model is not None and len(state.model.classes) >= 2
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# TYPE oncoai_samples_analysed_total counter",
            f"oncoai_samples_analysed_total {state.analysed_samples}",
            "# TYPE oncoai_predictions_abstained_total counter",
            f"oncoai_predictions_abstained_total {state.abstained}",
            "# TYPE oncoai_model_classes gauge",
            f"oncoai_model_classes {len(state.model.classes)}",
            "# TYPE oncoai_http_requests_total counter",
            *[
                f'oncoai_http_requests_total{{method="{k.split()[0]}",status="{k.split()[1]}"}} {v}'
                for k, v in sorted(state.counters.items())
            ],
        ]
        return "\n".join(lines) + "\n"

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        html = resources.files("oncoai").joinpath("dashboard/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html, headers={"content-security-policy": CSP})

    @app.get("/v1/whoami")
    def whoami(role: str = Depends(require("viewer"))) -> dict:
        return {"role": role}

    @app.get("/v1/model")
    def model_info(_: str = Depends(require("viewer"))) -> dict:
        return model_card(state.model)

    @app.get("/v1/cohort")
    def cohort(_: str = Depends(require("viewer"))) -> dict:
        return {"notice": RESEARCH_USE, "labels": state.model.meta.get("cohort", {})}

    @app.get("/v1/example", response_class=PlainTextResponse)
    def example(_: str = Depends(require("analyst"))) -> str:
        if not state.example:
            raise HTTPException(404, "no example file configured (ONCOAI_EXAMPLE_MAF)")
        return state.example

    @app.post("/v1/analyse")
    async def analyse(request: Request, role: str = Depends(require("analyst"))) -> dict:
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in TEXT_TYPES:
            raise HTTPException(415, f"send the MAF as {' or '.join(TEXT_TYPES)}")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            audit(request, role, "analyse", "rejected: too large", bytes=int(declared))
            raise HTTPException(413, f"file larger than {MAX_BODY_BYTES // (1024 * 1024)} MB")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                audit(request, role, "analyse", "rejected: too large", bytes=len(body))
                raise HTTPException(413, f"file larger than {MAX_BODY_BYTES // (1024 * 1024)} MB")
        digest = hashlib.sha256(body).hexdigest()
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            audit(request, role, "analyse", "rejected: not UTF-8", sha256=digest)
            raise HTTPException(422, "file is not UTF-8 text") from None
        params = request.query_params
        try:
            filters = Filters(min_vaf=float(params.get("min_vaf", 0)), min_alt_reads=int(params.get("min_alt_reads", 0)))
        except ValueError:
            raise HTTPException(422, "min_vaf must be a number and min_alt_reads an integer") from None
        if not (0 <= filters.min_vaf < 1) or not (0 <= filters.min_alt_reads <= 1000):
            raise HTTPException(422, "min_vaf must be in [0, 1) and min_alt_reads in [0, 1000]")
        try:
            data = read_maf(text, filters, max_rows=MAX_ROWS)
        except MafError as exc:
            audit(request, role, "analyse", f"rejected: {exc}", sha256=digest)
            raise HTTPException(422, f"not a usable MAF: {exc}") from None
        by_sample = data.by_sample()
        if len(by_sample) > MAX_SAMPLES:
            audit(request, role, "analyse", "rejected: too many samples", sha256=digest, samples=len(by_sample))
            raise HTTPException(422, f"at most {MAX_SAMPLES} samples per request")
        items = list(profiles(by_sample).values())
        preds = state.model.predict(items) if items else []
        state.analysed_samples += len(items)
        state.abstained += sum(p.abstained for p in preds)
        audit(request, role, "analyse", "ok", sha256=digest, samples=len(items), abstained=sum(p.abstained for p in preds))
        context = state.model.meta.get("cohort", {})
        results = []
        known_genes, known_changes = set(state.model.space.genes), set(state.model.space.changes)
        for p, pred in zip(items, preds, strict=True):
            results.append(
                {
                    "profile": p.summary(),
                    "prediction": pred.as_dict(),
                    "model_genes_hit": sorted(p.genes & known_genes),
                    "model_changes_hit": sorted(p.changes & known_changes),
                }
            )
        return {
            "notice": RESEARCH_USE,
            "qc": data.qc.as_dict(),
            "samples": results,
            "cohort_tmb": {k: v["tmb"] for k, v in context.items()},
        }

    @app.post("/v1/admin/reload")
    def reload(request: Request, role: str = Depends(require("admin"))) -> dict:
        if state.model_path is None:
            raise HTTPException(409, "the server runs the built-in synthetic demo model; set ONCOAI_MODEL to reload from a file")
        try:
            state.model = Model.load(state.model_path)
        except (OSError, ValueError, KeyError) as exc:
            audit(request, role, "reload", f"failed: {exc}")
            raise HTTPException(422, f"model not reloaded: {exc}") from None
        audit(request, role, "reload", "ok", classes=len(state.model.classes))
        return {"reloaded": True, "classes": state.model.classes}

    @app.get("/v1/admin/audit")
    def audit_trail(_: str = Depends(require("admin"))) -> dict:
        return {"entries": list(state.audit)}

    return app


def demo_model() -> tuple[Model, str]:
    """A small model trained on synthetic data, so the console works without downloads."""
    from .synthetic import label_of, maf_text
    from .training import train

    data = read_maf(maf_text(per_type=40, seed=1))
    items = list(profiles(data.by_sample()).values())
    model, _ = train(items, [label_of(p.sample) for p in items], [p.sample for p in items], n_splits=5, seed=7, target_accuracy=0.95)
    model.meta["synthetic"] = True
    example = maf_text(per_type=1, seed=99)
    return model, example
