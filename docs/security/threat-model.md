# Threat model (STRIDE)

## Assets
- Uploaded mutation files (possibly personal data if from non-public cohorts).
- The model file (integrity: a tampered model gives wrong answers with full confidence).
- API keys; the audit trail.
- Downloaded public data (integrity: tampered data poisons training).

## Trust boundaries
1. Browser / client ↔ API (keys, uploads).
2. API ↔ model file on disk (read at start and on admin reload).
3. Build environment ↔ cBioPortal datahub over the internet (`oncoai fetch`).

## Threats and mitigations

| STRIDE | Threat | Mitigation | Residual risk |
|---|---|---|---|
| Spoofing | Use of a stolen or guessed key | Keys ≥ 24 characters, stored only as SHA-256, compared in constant time; three roles with least privilege | Static keys have no expiry; use an identity provider in production (deployment guide) |
| Tampering | Replaced model file runs code on load | JSON model format, never pickle (ADR 0001); size, shape and finiteness checks | A valid-looking but wrong model is still accepted: protect the file with read-only mounts and review model changes |
| Tampering | Poisoned or swapped download | SHA-256 and size from the pinned manifest; temporary file moved into place only after both checks; HTTPS only | Trust in the datahub commit itself |
| Repudiation | Analyst denies an upload | Audit entry per analysis: time, request ID, role, SHA-256 of the body, sample count, outcome | Roles, not named users; in-memory trail is lost on restart unless logs are shipped |
| Information disclosure | Uploaded data kept or leaked | Processed in memory, never written; audit holds no content; `cache-control: no-store`; no CORS; console sends data only to its own origin (CSP `connect-src 'self'`) | Reverse proxies or APM agents could log bodies: configure them not to |
| Information disclosure | Direct identifiers in sample IDs | Sample IDs restricted to `[A-Za-z0-9._:-]{1,64}`; others rejected | Pseudonymous IDs can still be linkable; pseudonymise before upload |
| Information disclosure | Cross-site scripting through file content in the console | Console builds DOM with `textContent` and SVG attributes only; CSP `default-src 'self'`, `frame-ancestors 'none'`; test guards against `innerHTML` with data | `'unsafe-inline'` scripts are allowed for the single-file console; a build step with nonces would remove that |
| Denial of service | Huge or pathological uploads | 5 MB body limit (header and streamed), 100,000 rows, 25 samples per request; linear-time parsing | No rate limiting in the app: put it at the ingress |
| Elevation of privilege | Viewer analyses data or reloads model | Role checks on every route; tests for 401 and 403 on each role boundary | — |

## Out of scope
Host hardening, TLS termination, network policy and secrets storage belong to the platform that
runs the container (see the deployment guide).
