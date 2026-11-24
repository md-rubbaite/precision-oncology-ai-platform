# Data governance

## Public data used here
| Item | Detail |
|---|---|
| Source | cBioPortal datahub, TCGA PanCancer Atlas 2018 studies, pinned commit |
| Nature | Somatic mutation calls and sample annotations, de-identified by TCGA, open-access tier |
| Licences | Datahub: Open Database License (ODbL 1.0). TCGA studies: terms in each study folder, which ask users to treat the data as preliminary |
| Redistribution | None. The repository holds a manifest (file names, SHA-256, sizes), not the data. Downloaded files live in `data/`, which is git-ignored |
| Derived files | `model.json` contains coefficients and cohort summaries (TMB quantiles, gene frequencies). Treat it as derived from the datahub: attribute cBioPortal and TCGA if you share it, and check whether ODbL share-alike applies to your use |
| Attribution | Cite cBioPortal (Cerami et al., 2012; Gao et al., 2013), the MC3 call set (Ellrott et al., 2018) and the TCGA studies used |

TCGA's controlled-access tier (raw reads, germline variants) is not used and must not be loaded
into this platform.

## Uploaded data
- The service accepts MAF text only, analyses it in memory and returns the result. Nothing is written
  to disk, cached or used for training.
- The audit trail keeps: time, request ID, role, action, outcome, SHA-256 of the body, sample count.
  It never keeps mutations, gene names or sample IDs.
- Sample IDs must match `[A-Za-z0-9._:-]{1,64}`; rows with other IDs (spaces, @, names) are rejected
  and counted, so direct identifiers are less likely to slip through. This is a guard, not
  de-identification: pseudonymise before upload.
- Somatic variants can still be identifying in combination with other data. Treat uploads from
  non-public cohorts as personal data.

## If this were used with non-public data
- Legal basis and ethics approval for the research use (for example GDPR Article 9(2)(j) with
  national safeguards, or the HIPAA research provisions in the United States).
- Data protection impact assessment; data processing agreement with whoever hosts the service.
- Hosting in the region the approval requires; encryption in transit (TLS at the ingress) and at rest
  for any logs.
- Access by named users through an identity provider, least privilege, periodic access review.
- Log retention aligned with the approval; audit export to the institution's SIEM.
