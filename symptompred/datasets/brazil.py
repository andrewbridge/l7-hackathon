"""Brazil e-SUS Notifica flu-syndrome notifications (via scripts/download_brazil.py).

Source: Ministério da Saúde open-data API (notificacoes-de-sindrome-gripal-leve-<year>).
Data dictionary: https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SGL/pdfs/dicionario-de-dados_e-sus-notifica-opendatasus-1.pdf

Mapping notes:
  * A notification can record up to 4 tests (test 1 = most recent). We take the first one of an
    accepted type (default RT-PCR only) with a conclusive result (1 = detectable, 2 = not detectable).
  * `date` = that test's collection date, falling back to the notification date when it's missing
    or implausible (the raw data contains values like 1970-01-01, or years after the notification).
  * Symptoms come from the comma-separated `sintomas` field. "Assintomático" means all symptoms are 0.
    If `sintomas` is blank, symptoms are NaN (not recorded).
  * There is no contact-with-a-case field, so `contact` is absent.

Representativeness (checked Oct 2026 on the full 2020–2022 API download, 600,366 notifications):
  * ~98% of rows are from one state, Acre. The API serves a slice of the national data, not all of it.
  * Ages are effectively capped in the mid-50s (none ≥ 60 in 2020, one in 2021), so `age_60` carries
    almost no information here. Treat this as "under-55s tested in Acre", not "Brazil".
  * The notification system is for flu-like illness, so nearly everyone in it is symptomatic.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import PROJECT_ROOT
from ..schema import validate

DATA_DIR = PROJECT_ROOT / "data" / "brazil"
TEST_TYPES = {"1": "RT-PCR", "2": "RT-LAMP", "3": "rapid antigen", "4": "rapid antibody IgM",
              "5": "rapid antibody IgG", "6": "serology IgA", "7": "serology IgM", "8": "serology IgG",
              "9": "serology total antibodies"}
SYMPTOMS = {  # raw value in `sintomas` -> schema name
    "Tosse": "cough", "Febre": "fever", "Dor de Garganta": "sore_throat", "Dispneia": "shortness_of_breath",
    "Dor de Cabeça": "headache", "Distúrbios Olfativos": "loss_of_smell",
    "Distúrbios Gustativos": "loss_of_taste", "Coriza": "runny_nose",
}
# Brazil's first confirmed case to the end of the downloaded data; a handful of rows fall outside
VALID_DATES = ("2020-02-26", "2022-12-31")
MAX_DATE_GAP_DAYS = 30


USED_COLUMNS = (
    "excluido", "sintomas", "sexo", "idade", "estado", "data_notificacao", "classificacao_final",
    *(f"{field}_{k}" for k in range(1, 5)
      for field in ("codigo_tipo_teste", "codigo_resultado_teste", "data_coleta_teste")),
)


def available(path: str | Path | None = None) -> bool:
    """Whether any downloaded pages exist."""
    root = Path(path or DATA_DIR)
    return root.exists() and any(root.glob("*/offset_*.json.gz"))


def read_raw(path: str | Path | None = None, years=None, columns=USED_COLUMNS) -> pd.DataFrame:
    """Read the downloaded pages (all years, or just `years`) into one DataFrame of raw fields.

    Only `columns` are kept (default: those `load` uses), which keeps memory manageable.
    Pass columns=None for every field. Adds a `file_year` column (the API dataset each row came from).
    """
    root = Path(path or DATA_DIR)
    dirs = [root / str(y) for y in years] if years else sorted(p for p in root.iterdir() if p.is_dir())
    frames = []
    for d in dirs:
        for f in sorted(d.glob("offset_*.json.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                page = pd.DataFrame(json.load(fh))
            if columns is not None:
                page = page.reindex(columns=list(columns))
            frames.append(page.assign(file_year=int(d.name)))
    if not frames:
        raise FileNotFoundError(f"No downloaded pages under {root}. Run scripts/download_brazil.py first.")
    return pd.concat(frames, ignore_index=True)


def _valid_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return d.where(d.between(*VALID_DATES))


def load(path: str | Path | None = None, years=None, test_types=("1",), raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load as the symptompred schema. `test_types` are codes from TEST_TYPES (default RT-PCR only).

    Pass `raw` (from `read_raw`) to reuse data already in memory instead of reading the files again.
    """
    raw = read_raw(path, years) if raw is None else raw
    raw = raw[raw["excluido"] != "true"].reset_index(drop=True)

    # First test (most recent first) of an accepted type with a conclusive result
    result = pd.Series(np.nan, index=raw.index)
    test_date = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    for k in range(1, 5):
        ok = (raw[f"codigo_tipo_teste_{k}"].isin(test_types)
              & raw[f"codigo_resultado_teste_{k}"].isin(["1", "2"]) & result.isna())
        result[ok] = (raw.loc[ok, f"codigo_resultado_teste_{k}"] == "1").astype(float)
        test_date[ok] = _valid_date(raw.loc[ok, f"data_coleta_teste_{k}"])
    keep = result.notna()
    raw, result, test_date = raw[keep], result[keep], test_date[keep]

    # Trust the collection date only if it's within MAX_DATE_GAP of the notification date
    notified = _valid_date(raw["data_notificacao"])
    test_date = test_date.where((test_date - notified).abs() <= pd.Timedelta(days=MAX_DATE_GAP_DAYS))
    date = test_date.fillna(notified)
    has_date = date.notna()
    raw, result, date = raw[has_date], result[has_date], date[has_date]

    reported = raw["sintomas"].fillna("").str.split(",").apply(lambda xs: {x.strip() for x in xs if x.strip()})
    recorded = raw["sintomas"].notna()
    symptoms = {name: reported.apply(lambda s, v=value: float(v in s)).where(recorded)
                for value, name in SYMPTOMS.items()}

    age = pd.to_numeric(raw["idade"], errors="coerce")
    age = age.where(age.between(0, 120))
    df = pd.DataFrame({
        "date": date,
        "y": result.astype(int),
        **symptoms,
        "age_60": (age >= 60).astype(float).where(age.notna()),
        "male": raw["sexo"].map({"Masculino": 1, "Feminino": 0}),
        "age": age,
        "region": raw["estado"],
        "source": "brazil",
    })
    return validate(df.reset_index(drop=True))
