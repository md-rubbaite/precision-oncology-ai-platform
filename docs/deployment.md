# Deployment

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `ONCOAI_API_KEYS` | `role:key` pairs, comma separated; roles `viewer`, `analyst`, `admin`; keys ≥ 24 characters | required |
| `ONCOAI_MODEL` | Path to a trained `model.json` | built-in synthetic demo model |
| `ONCOAI_EXAMPLE_MAF` | File offered by the console's Example button | none (button returns 404) |
| `ONCOAI_HOST`, `ONCOAI_PORT` | Bind address | `127.0.0.1`, `8080` (container: `0.0.0.0`) |

## Train once, serve many

```bash
oncoai fetch --studies benchmark --dest data/pancan
oncoai train --data data/pancan --out models/model.json --report models/report.json
# review models/report.json, then
docker compose up -d --build        # mounts ./models read-only, see docker-compose.yml
```

Training needs scikit-learn (`pip install ".[train]"`). Serving needs only NumPy and FastAPI; the
image installs the `train` extra so the demo model works without a file.

## Container
- Non-root user (UID 10001), read-only root file system, `/tmp` as tmpfs, all capabilities dropped,
  `no-new-privileges`.
- Health check on `/healthz`; readiness on `/readyz`.
- Published only on `127.0.0.1` in `docker-compose.yml`: put a TLS-terminating reverse proxy in front.

## Production checklist
- [ ] TLS at the ingress; HSTS.
- [ ] Rate limiting and request-size limits at the ingress as well (defence in depth).
- [ ] Replace static keys with an identity provider (OIDC) in front of the service, or rotate keys
      through a secrets manager; never commit `.env`.
- [ ] Ship JSON logs (access and audit) to central logging; make sure no component logs request bodies.
- [ ] Mount `model.json` read-only; change models through review, then `POST /v1/admin/reload`.
- [ ] Scrape `/metrics`; alert on `oncoai_predictions_abstained_total` jumping (data drift) and 5xx rates.
- [ ] Data protection: see [data governance](data/data-governance.md) before using non-public data.
