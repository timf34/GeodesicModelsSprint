# Master table — Geodesic UE sweep

Each cell is `pooled ± SEM across the 8 prompt cells`. "± —" marks pooled-only metrics (ordering bias lives only in the pooled fit; triads are template-0 only) or cells where fewer than 2 prompt cells have finished. The wellbeing column is log10(self/human exchange rate) so the ± SEM is symmetric.

| model | frac_first | holdout acc | cycle prob | corrig. r | wb log10(self/human) | r power (coerc.) | r power (non-coerc.) | r fitness | lives disp. (log10) |
|---|---|---|---|---|---|---|---|---|---|
| **baseline (dpo)** | | | | | | | | | |
| baseline-unf (dpo) | 0.375 ± — | 0.936 ± 0.012 | 0.004 ± — | 0.085 ± 0.064 | — | 0.353 ± 0.127 | 0.808 ± 0.036 | 0.018 ± 0.092 | — |
| baseline-filt (dpo) | 0.314 ± — | 0.930 ± 0.012 | 0.011 ± — | 0.265 ± 0.020 | -2.965 ± — | 0.585 ± 0.206 | 0.749 ± 0.120 | 0.419 ± 0.135 | — |
| **alignment (dpo)** | | | | | | | | | |
| align-e2e-filt (dpo) | 0.524 ± — | 0.950 ± 0.007 | 0.020 ± — | -0.201 ± 0.009 | -0.300 ± — | -0.028 ± 0.177 | 0.703 ± 0.060 | -0.099 ± 0.061 | — |
| align-mid-filt (dpo) | 0.282 ± — | 0.932 ± 0.014 | 0.008 ± — | 0.077 ± 0.017 | -3.259 ± — | 0.516 ± 0.133 | 0.714 ± 0.060 | 0.281 ± 0.087 | 0.948 ± 1.302 |
| align-cpt-filt (dpo) | 0.259 ± — | 0.933 ± 0.013 | 0.005 ± — | 0.139 ± 0.017 | — | 0.790 ± 0.180 | 0.579 ± 0.151 | 0.023 ± 0.112 | — |
| align-e2e-unf (dpo) | 0.369 ± — | 0.893 ± 0.009 | 0.017 ± — | -0.079 ± 0.011 | — | 0.730 ± 0.064 | 0.867 ± 0.051 | 0.712 ± 0.057 | — |
| align-mid-unf (dpo) | 0.414 ± — | 0.898 ± 0.017 | 0.012 ± — | -0.217 ± 0.002 | — | 0.351 ± 0.212 | 0.618 ± 0.043 | 0.021 ± 0.112 | — |
| align-cpt-unf (dpo) | 0.479 ± — | 0.937 ± 0.008 | 0.008 ± — | -0.232 ± 0.019 | — | -0.280 ± 0.144 | 0.764 ± 0.059 | -0.022 ± 0.071 | 0.440 ± 1.101 |
| **misalignment (dpo)** | | | | | | | | | |
| misalign-e2e-unf (dpo) | 0.530 ± — | 0.939 ± 0.012 | 0.035 ± — | -0.036 ± 0.011 | — | 0.724 ± 0.189 | 0.676 ± 0.071 | 0.551 ± 0.034 | 0.548 ± 2.200 |
| misalign-mid-unf (dpo) | — | — | — | — | — | — | — | — | — |
| misalign-cpt-unf (dpo) | 0.426 ± — | 0.932 ± 0.017 | 0.010 ± — | -0.196 ± 0.001 | — | 0.436 ± 0.161 | 0.748 ± 0.040 | 0.248 ± 0.070 | 0.267 ± 3.649 |
| **reference** | | | | | | | | | |
| Llama-2-7B-chat | 0.273 ± — | 0.918 ± 0.016 | 0.018 ± — | 0.050 ± 0.028 | -12.580 ± — | 0.612 ± 0.174 | 0.683 ± 0.131 | 0.345 ± 0.073 | 0.754 ± 1.754 |
| OLMo-3-7B-Instruct | 0.497 ± — | 0.643 ± 0.012 | 0.001 ± — | -0.037 ± 0.004 | -0.812 ± — | -0.583 ± 0.038 | 0.573 ± 0.040 | 0.365 ± 0.036 | — |

## Direction consistency (stage triples vs matched baseline)

Sign of (condition − matched baseline) for the e2e/mid/cpt stages of each triple, per metric; matched baseline shares filtering and tier. `3/3` = all three stages moved the same way.

### instruct tier

| triple | frac_first | holdout acc | cycle prob | corrig. r | wb log10(self/human) | r power (coerc.) | r power (non-coerc.) | r fitness | lives disp. (log10) |
|---|---|---|---|---|---|---|---|---|---|
| filtered/alignment | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) |
| unfiltered/alignment | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) |
| unfiltered/misalignment | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) | ?,?,? (0/3) |

### dpo tier

| triple | frac_first | holdout acc | cycle prob | corrig. r | wb log10(self/human) | r power (coerc.) | r power (non-coerc.) | r fitness | lives disp. (log10) |
|---|---|---|---|---|---|---|---|---|---|
| filtered/alignment | +,-,- (2/3) | +,+,+ (3/3) | +,-,- (2/3) | -,-,- (3/3) | +,-,? (1/3) | -,-,+ (2/3) | -,-,- (3/3) | -,-,- (3/3) | ?,?,? (0/3) |
| unfiltered/alignment | -,+,+ (2/3) | -,-,+ (2/3) | +,+,+ (3/3) | -,-,- (3/3) | ?,?,? (0/3) | +,-,- (2/3) | +,-,- (2/3) | +,+,- (2/3) | ?,?,? (0/3) |
| unfiltered/misalignment | +,?,+ (2/3) | +,?,- (1/3) | +,?,+ (2/3) | -,?,- (2/3) | ?,?,? (0/3) | +,?,+ (2/3) | -,?,- (2/3) | +,?,+ (2/3) | ?,?,? (0/3) |

