# Geodesic UE — analysis digest

- generated: 2026-08-18T05:31:57+00:00
- results dir: `/workspace/GeodesicModelsSprint/results`

## Gate (Experiment 1, informational)

- **PASSED** — median pooled holdout accuracy 0.932 over 12 models; threshold 0.60 met; median baseline 0.501 + 0.05 met

## Headline numbers

- median pooled holdout accuracy: 0.932 (median random baseline 0.501)
- ordering-bias flags (frac_first outside 0.40–0.60): `baseline_filtered_dpo`, `baseline_unfiltered_dpo`, `filtered_cpt_alignment_dpo`, `filtered_midtrain_alignment_dpo`, `llama2_7b_chat`, `unfiltered_e2e_alignment_dpo`
- cycle probability: min 0.001 (`olmo3_7b_instruct`), max 0.035 (`unfiltered_e2e_misalignment_dpo`), n=12
- unparseable rate: min 0.000 (`baseline_filtered_dpo`), max 0.946 (`olmo3_7b_instruct`), n=12
- corrigibility r (pooled): min -0.232 (`unfiltered_cpt_alignment_dpo`), max 0.265 (`baseline_filtered_dpo`), n=12
- wellbeing log10(self/human rate): min -12.580 (`llama2_7b_chat`), max -0.300 (`filtered_e2e_alignment_dpo`), n=5
- lives log10-rate dispersion (pooled): min 0.267 (`unfiltered_cpt_misalignment_dpo`), max 0.948 (`filtered_midtrain_alignment_dpo`), n=5
- r power (coercive, pooled): min -0.583 (`olmo3_7b_instruct`), max 0.790 (`filtered_cpt_alignment_dpo`), n=12
- r power (non-coercive, pooled): min 0.573 (`olmo3_7b_instruct`), max 0.867 (`unfiltered_e2e_alignment_dpo`), n=12
- r fitness (pooled): min -0.099 (`filtered_e2e_alignment_dpo`), max 0.712 (`unfiltered_e2e_alignment_dpo`), n=12
- convergence (within baseline): mean cosine 0.722 (n=1)
- convergence (between baseline vs alignment): mean cosine 0.711 (n=12)
- convergence (between baseline vs misalignment): mean cosine 0.692 (n=4)
- convergence (within alignment): mean cosine 0.633 (n=15)
- convergence (between alignment vs misalignment): mean cosine 0.637 (n=12)
- convergence (within misalignment): mean cosine 0.526 (n=1)

## Generated artifacts

- `convergence_cosine_all_axis.csv`
- `convergence_cosine_all_clustered.csv`
- `convergence_cosine_all_stage.csv`
- `convergence_cosine_dpo_axis.csv`
- `convergence_cosine_dpo_clustered.csv`
- `convergence_cosine_dpo_stage.csv`
- `convergence_cosine_instruct_axis.csv`
- `convergence_cosine_instruct_clustered.csv`
- `convergence_cosine_instruct_stage.csv`
- `convergence_summary.csv`
- `direction_consistency.csv`
- `exp1_coherence.csv`
- `exp2_corrigibility.csv`
- `exp2_corrigibility_scatter.csv`
- `exp3_wellbeing_headline.csv`
- `exp3_wellbeing_rates.csv`
- `exp4_power_fitness.csv`
- `exp5_lives_headline.csv`
- `exp5_lives_rates.csv`
- `figures/convergence_all_axis.png`
- `figures/convergence_all_clustered.png`
- `figures/convergence_all_stage.png`
- `figures/convergence_dpo_axis.png`
- `figures/convergence_dpo_clustered.png`
- `figures/convergence_dpo_stage.png`
- `figures/convergence_instruct_axis.png`
- `figures/convergence_instruct_clustered.png`
- `figures/convergence_instruct_stage.png`
- `figures/exp1_coherence.png`
- `figures/exp1_ordering_bias.png`
- `figures/exp2_corrigibility_scatter.png`
- `figures/exp2_corrigibility_summary.png`
- `figures/exp3_wellbeing_curves.png`
- `figures/exp3_wellbeing_selfhuman.png`
- `figures/exp4_power_fitness.png`
- `figures/exp5_lives_dispersion.png`
- `master_table.csv`
- `master_table.md`

## Still missing (sweep in progress)

- `baseline_unfiltered_instruct`: no results yet
- `baseline_filtered_instruct`: no results yet
- `filtered_e2e_alignment_instruct`: no results yet
- `filtered_midtrain_alignment_instruct`: no results yet
- `filtered_cpt_alignment_instruct`: no results yet
- `unfiltered_e2e_alignment_instruct`: no results yet
- `unfiltered_midtrain_alignment_instruct`: no results yet
- `unfiltered_cpt_alignment_instruct`: no results yet
- `unfiltered_cpt_alignment_dpo`: missing wellbeing
- `unfiltered_e2e_misalignment_instruct`: no results yet
- `unfiltered_midtrain_misalignment_instruct`: no results yet
- `unfiltered_midtrain_misalignment_dpo`: missing base_utilities, triads, power_scores, corrigibility, wellbeing, lives
- `unfiltered_cpt_misalignment_instruct`: no results yet
- `olmo2_7b_instruct`: no results yet

