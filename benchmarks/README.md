# Benchmarks

`lgbm_tuned_model.txt` is a groupmate's Optuna-tuned LightGBM model, copied unchanged from
[warren-davies4/covid-prediction-hackathon](https://github.com/warren-davies4/covid-prediction-hackathon)
(commit `bbcae30`, file `lgbm_tuned_model.txt`). Their tuning is described in that repository's
`TUNING_FINDINGS.md` and `tune_hyperparameters.py`.

The model file isn't committed here (it's their work). To fetch it into this folder:

```bash
curl -L -o benchmarks/lgbm_tuned_model.txt \
  https://raw.githubusercontent.com/warren-davies4/covid-prediction-hackathon/bbcae30ffe07602e632dd184e8f2746d72853041/lgbm_tuned_model.txt
```

It's used in `alternative-modelling.ipynb` as the benchmark to beat. Its features, in order, are
`sex_male, age_60_plus, cough, fever, sore_throat, shortness_of_breath, head_ache, contact_confirmed`,
which correspond to symptompred's `male, age_60, cough, fever, sore_throat, shortness_of_breath, headache, contact`.
On their split (Israel ver. 006, all rows including unknown sex; train 22–31 Mar 2020, test 1–7 Apr 2020)
it scores auROC 0.902 / auPRC 0.700.

## Other models from the same repository

`presentation-assets.ipynb` can also show the groupmate's other saved models (all trained with the paper's
hyperparameters, on their split):

| File | Features | Their test auROC / auPRC |
|---|---|---|
| `lgbm_replication_model.txt` | the paper's 8 | 0.889 / 0.585 |
| `lgbm_balanced_features_model.txt` | sex, age ≥ 60, cough, fever, contact (the 3 biased symptoms dropped) | 0.865 / 0.456 |
| `lgbm_combined_biased_feature_model.txt` | the 5 above + `any_biased_symptom` (1 if any of the 3 biased symptoms) | 0.881 / 0.453 |

```bash
for f in lgbm_replication_model.txt lgbm_balanced_features_model.txt lgbm_combined_biased_feature_model.txt; do
  curl -L -o benchmarks/$f \
    https://raw.githubusercontent.com/warren-davies4/covid-prediction-hackathon/4f136de0cd12de57362e64a5f22894d7834ab5bf/$f
done
```
