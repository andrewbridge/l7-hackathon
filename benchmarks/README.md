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
