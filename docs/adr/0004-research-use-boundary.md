# ADR 0004: Research-use boundary and data handling

- Status: accepted
- Date: 2026-11-24

## Context
Software that names a tumour type from DNA can be mistaken for a diagnostic. In the EU, software
intended to provide information for diagnosis falls under the medical device regulations (MDR /
IVDR); similar rules apply elsewhere. This project has no quality management system, no clinical
validation and no regulatory clearance. Public cancer data also comes with licence terms, and
uploaded research data may be personal data.

## Decision
- The intended purpose is research and teaching. Every analysis and model response, the model file, the CLI help
  and the console carry the notice "Research use only. Not for diagnosis, prognosis or treatment
  decisions."
- The platform gives no treatment, drug or trial recommendations and has no knowledge base of
  actionable variants.
- Public data is fetched by the user from the pinned source and never committed to the repository.
  Tests and the demo use invented data (types SYNA–SYNE, genes SG001–SG400).
- Uploads are analysed in memory and not stored; the audit trail records hashes and counts only;
  sample IDs that look like names or e-mail addresses are rejected.

## Consequences
- Anyone who wants to use the approach clinically has to start a regulated development process;
  this repository is at most a research reference for it.
- Users must cite cBioPortal, MC3 and TCGA when publishing results, and handle `model.json` as a
  work derived from the datahub.
- No persistence means no history of past analyses in the service; researchers keep their own records.
