# Roadmap

Research directions, in rough order. None of them changes the research-use boundary (ADR 0004).

1. **All 32 PanCancer types** once the remaining datahub files can be fetched; per-type results and a "type not represented" analysis.
2. **External validation** on an independent public cohort with a different calling pipeline.
3. **Centre-grouped folds** to measure sequencing-centre effects within TCGA.
4. **Panel-aware features**: TMB and gene features restricted to a panel's footprint, so panel data can be scored against an exome-trained model fairly.
5. **DISCOVER-style pair test** that accounts for per-tumour mutation rates.
6. **Copy-number features** (arm-level) where available; measured as an ablation like SBS96.
7. **OIDC** in front of the API and per-user audit.
