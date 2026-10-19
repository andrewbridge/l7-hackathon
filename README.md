# Predicting COVID-19 test results from symptoms

A hackathon project that reproduces and stress-tests
[Zoabi et al. (2021), *Machine learning-based prediction of COVID-19 diagnosis based on symptoms*](https://www.nature.com/articles/s41746-020-00372-6)
(npj Digital Medicine). The paper used a LightGBM model on 8 yes/no answers (symptoms, age over 60, sex,
contact with a confirmed case) to predict a positive PCR test in Israel.

We ask three questions, one notebook each:

| Notebook | Question | Headline |
|---|---|---|
| [`notebook.ipynb`](notebook.ipynb) | Does the paper reproduce, and what's driving it? | It reproduces (auROC 0.90), but symptoms were recorded mostly for people who tested positive (reporting bias), and contact with a case carries much of the signal. |
| [`alternative-data.ipynb`](alternative-data.ipynb) | Does the approach hold up on other countries' data? | Not really: 0.71 in the US (Carbon Health) and 0.64 in Brazil, where symptoms were recorded before the result was known. |
| [`alternative-modelling.ipynb`](alternative-modelling.ipynb) | Can a different technique do better? | Several beat the paper's model, but none beats a tuned LightGBM. With 8 answers there's almost no headroom, and what the model is given matters far more than which model it is. |

Slide-ready charts are in [`assets/`](assets/).

## What's in the repository

| Path | Contents |
|---|---|
| `symptompred/` | A small library (in the spirit of Prophet): one data convention for every dataset, loaders, the paper's model, alternative models, and evaluation and plotting tools. `help(symptompred)` is the starting point. |
| `scripts/download_brazil.py` | A polite, resumable downloader for the Brazilian data (see below). |
| `tests/` | `pytest` suite. Fast tests use synthetic data. Slow tests use the real datasets and pin the notebooks' headline numbers. |
| `assets/` | PNG charts exported by the notebooks. |
| `benchmarks/` | Instructions for fetching a groupmate's tuned model (the benchmark in `alternative-modelling.ipynb`). |
| `requirements.txt`, `devbox.json` | Python dependencies, and an optional [Devbox](https://www.jetify.com/devbox) environment. |

## Setup

Python 3.14 (any recent 3.x should work).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Or, with Devbox: `devbox shell` (it activates `.venv` for you), then the `pip install` line above.

## Data and files not included

The datasets, the original authors' code and a groupmate's model aren't ours to redistribute, so they're
excluded by `.gitignore`. Put them back at these paths (relative to the repository root) to run everything:

| What | Needed by | Where it goes | How to get it |
|---|---|---|---|
| **Israeli Ministry of Health data + the paper's code and models** | all three notebooks | `covidpred-master/` (must contain `data/corona_tested_individuals_ver_006.english.csv.zip`) | `git clone https://github.com/nshomron/covidpred covidpred-master` |
| **US: Carbon Health clinical data** (CC BY-NC-SA 4.0) | `alternative-data`, `alternative-modelling` | `covidclinicaldata-master/data/*.csv` (29 weekly CSV files) | `git clone https://github.com/mdcollab/covidclinicaldata covidclinicaldata-master` |
| **Brazil: e-SUS Notifica flu-syndrome notifications** (about 600k rows, 37 MB) | `alternative-data`, `alternative-modelling` | `data/brazil/2020/`, `2021/`, `2022/` (`offset_*.json.gz` pages) | `python scripts/download_brazil.py`, see below |
| **Groupmate's tuned LightGBM** | `alternative-modelling` | `benchmarks/lgbm_tuned_model.txt` | see [`benchmarks/README.md`](benchmarks/README.md) |
| The paper (PDF) | reference only | anywhere | [open access from Nature](https://www.nature.com/articles/s41746-020-00372-6.pdf) |

Expected layout once everything is in place:

```
.
├── covidpred-master/
│   ├── data/corona_tested_individuals_ver_006.english.csv.zip
│   ├── hyperparameters.txt
│   └── lgbm_model_*.txt
├── covidclinicaldata-master/
│   └── data/04-07_carbonhealth_and_braidhealth.csv ...
├── data/
│   └── brazil/{2020,2021,2022}/offset_*.json.gz
├── benchmarks/lgbm_tuned_model.txt
├── symptompred/ ...
└── *.ipynb
```

If your data lives elsewhere, every loader takes a `path` argument (for example
`sp.datasets.brazil.load(path=...)`), and `alternative-data.ipynb` also reads a `BRAZIL_DIR`
environment variable.

### Downloading the Brazilian data

The bulk CSV files on Brazil's open-data portal return 403, so the script pages through the Ministry of
Health's public API instead. It's deliberately gentle: one request at a time, at least 3 seconds apart
(longer if the server is slow), with regular rests and backoff on errors. It's also resumable: rerun the
same command to continue after a stop. A full download is about 600 requests and takes roughly an hour.

```bash
python scripts/download_brazil.py                              # 2020–2022 into data/brazil/
python scripts/download_brazil.py --years 2020 --max-pages 10  # quick trial
python scripts/download_brazil.py --help                       # all options
```

Note that the API serves a slice of the national data: about 98% of rows are from one state (Acre), and
ages are effectively capped in the mid-50s. See the docstring of `symptompred/datasets/brazil.py`.

## Running

- **Notebooks:** open in Jupyter or VS Code with the `.venv` kernel and run all cells. `alternative-modelling.ipynb`
  takes about 10 minutes and the others a few minutes each. They write their charts to `assets/`.
- **Tests:**
  ```bash
  python -m pytest -m "not slow"   # about 10 seconds, no data needed
  python -m pytest                 # adds real-data regression tests (about 1–2 minutes)
  ```
  Slow tests are skipped automatically if the data isn't present (Brazil-specific ones need `data/brazil/`).

## Using the library

```python
import symptompred as sp

israel = sp.datasets.israel.load(drop_unknown_sex=True)
train, test = sp.date_split(israel, sp.datasets.israel.PAPER_TRAIN, sp.datasets.israel.PAPER_TEST)

model = sp.SymptomModel().fit(train)              # the paper's LightGBM
model.evaluate(test, sensitivity=0.873)           # auROC, auPRC, CIs, working point

lr = sp.registry()["Logistic + interactions"]().fit(train)   # any of 15 alternative techniques
sp.paired_bootstrap(test.y, lr.predict(test), model.predict(test))
```

To add a dataset, write a loader that returns a DataFrame with `date`, `y` and feature columns named from
`sp.ALL_FEATURES`, and pass it through `sp.validate`.

## Credits and licences

The code in this repository is MIT licensed (see [`LICENSE`](LICENSE)). The datasets and third-party code it uses keep their own terms:

- Zoabi, Deri-Rozov & Shomron (2021), npj Digital Medicine 4:3, and their code and data at
  [nshomron/covidpred](https://github.com/nshomron/covidpred). The data is from the Israeli Ministry of Health.
- [Carbon Health COVID-19 Clinical Data Repository](https://github.com/mdcollab/covidclinicaldata), CC BY-NC-SA 4.0.
- Brazil: Ministério da Saúde, [Portal de Dados Abertos do SUS](https://opendatasus.saude.gov.br/), e-SUS Notifica.
- Tuned benchmark model: [warren-davies4/covid-prediction-hackathon](https://github.com/warren-davies4/covid-prediction-hackathon).
