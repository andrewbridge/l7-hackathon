import gzip
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from symptompred.datasets import brazil, carbon_health, israel


# ------------------------------------------------------------------ Israel
def _israel_zip(path):
    raw = pd.DataFrame({
        "test_date": ["2020-03-22", "2020-03-23", "2020-03-24", "2020-03-25", "2020-03-26"],
        "cough": [1, 0, 0, 1, 0], "fever": [0, 1, 0, 1, 0], "sore_throat": [0, 0, 0, 0, 1],
        "shortness_of_breath": [0, 0, 1, 0, 0], "head_ache": [1, 0, 0, 0, 0],
        "corona_result": ["positive", "negative", "other", "negative", "positive"],
        "age_60_and_above": ["Yes", "No", "No", None, "No"],
        "gender": ["male", "female", "male", None, "female"],
        "test_indication": ["Contact with confirmed", "Other", "Abroad", "Other", "Contact with confirmed"],
    })
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("__MACOSX/._data.csv", b"\x00\x05binary resource fork")  # must be skipped
        z.writestr("data.csv", raw.to_csv(index=False))
    return path


def test_israel_load(tmp_path):
    df = israel.load(_israel_zip(tmp_path / "il.csv.zip"))
    assert len(df) == 4  # "other" result dropped
    assert df["y"].tolist() == [1, 0, 0, 1]
    assert df["contact"].tolist() == [1, 0, 0, 1]
    assert df["headache"].tolist() == [1, 0, 0, 0]
    assert df["male"].tolist()[:2] == [1, 0] and np.isnan(df["male"].iloc[2])
    assert np.isnan(df["age_60"].iloc[2])
    assert (df["source"] == "israel").all()


def test_israel_drop_unknown_sex(tmp_path):
    df = israel.load(_israel_zip(tmp_path / "il.csv.zip"), drop_unknown_sex=True)
    assert len(df) == 3 and df["male"].notna().all()


# ----------------------------------------------------------- Carbon Health
def _carbon_dir(path):
    path.mkdir()
    base = {c: [True, False, False] for c in carbon_health.SYMPTOMS}
    pd.DataFrame({"batch_date": ["2020-04-07"] * 3, "covid19_test_results": ["Positive", "Negative", "Negative"],
                  "age": [65, -1, 30], "test_name": ["SARS-CoV-2, NAA", "Rapid COVID-19 Test", "SARS-CoV-2, NAA"],
                  "high_risk_interactions": [True, None, False], **base}).to_csv(path / "2020-04-07.csv", index=False)
    pd.DataFrame({"batch_date": ["2020-04-14"] * 2, "covid19_test_results": ["Negative", "Unknown"],
                  "age": [70, 20], "test_name": ["SARS-CoV-2, NAA"] * 2,
                  "high_risk_interactions": [False, True], **{c: [False, True] for c in carbon_health.SYMPTOMS},
                  }).to_csv(path / "2020-04-14.csv", index=False)
    return path


def test_carbon_health_load(tmp_path):
    df = carbon_health.load(_carbon_dir(tmp_path / "ch"))
    assert len(df) == 4  # "Unknown" result dropped
    assert np.isnan(df["age"].iloc[1]) and np.isnan(df["age_60"].iloc[1])  # -1 placeholder -> NaN
    assert df["age_60"].iloc[[0, 2, 3]].tolist() == [1, 0, 1]
    assert np.isnan(df["contact"].iloc[1])
    assert "male" not in df.columns
    assert {"loss_of_smell", "muscle_ache", "shortness_of_breath"} <= set(df.columns)


def test_carbon_health_molecular_only(tmp_path):
    assert len(carbon_health.load(_carbon_dir(tmp_path / "ch"), molecular_only=True)) == 3


def test_carbon_health_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        carbon_health.load(tmp_path)


# ----------------------------------------------------------------- Brazil
def _row(**kw):
    row = {c: None for c in brazil.USED_COLUMNS}
    row.update({"excluido": "false", "sexo": "Masculino", "idade": "40", "estado": "Acre",
                "data_notificacao": "2021-05-10"})
    row.update(kw)
    return row


BRAZIL_ROWS = [
    # 0: RT-PCR positive, valid collection date, symptoms
    _row(sintomas="Tosse, Febre, Distúrbios Olfativos", codigo_tipo_teste_1="1", codigo_resultado_teste_1="1",
         data_coleta_teste_1="2021-05-08"),
    # 1: antigen only -> dropped with the default RT-PCR filter
    _row(sintomas="Tosse", codigo_tipo_teste_1="3", codigo_resultado_teste_1="1", data_coleta_teste_1="2021-05-08"),
    # 2: inconclusive test 1, conclusive RT-PCR test 2 with an implausible date -> notification date
    _row(sintomas="Assintomático", sexo="Feminino", idade="70", codigo_tipo_teste_1="1", codigo_resultado_teste_1="3",
         codigo_tipo_teste_2="1", codigo_resultado_teste_2="2", data_coleta_teste_2="1970-01-01"),
    # 3: collection date valid but >30 days from notification -> notification date; blank symptoms
    _row(sintomas=None, codigo_tipo_teste_1="1", codigo_resultado_teste_1="2", data_coleta_teste_1="2021-01-01"),
    # 4: deleted record
    _row(excluido="true", sintomas="Tosse", codigo_tipo_teste_1="1", codigo_resultado_teste_1="1",
         data_coleta_teste_1="2021-05-08"),
]


def _brazil_dir(path):
    (path / "2021").mkdir(parents=True)
    with gzip.open(path / "2021" / "offset_00000000.json.gz", "wt", encoding="utf-8") as f:
        json.dump(BRAZIL_ROWS, f)
    return path


def test_brazil_load(tmp_path):
    root = _brazil_dir(tmp_path / "br")
    assert brazil.available(root)
    df = brazil.load(root)
    assert len(df) == 3
    assert df["y"].tolist() == [1, 0, 0]
    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2021-05-08", "2021-05-10", "2021-05-10"]
    assert df.loc[0, ["cough", "fever", "loss_of_smell", "headache"]].tolist() == [1, 1, 1, 0]
    # "Assintomático" = all symptoms recorded as absent; blank = not recorded
    assert (df.loc[1, list(brazil.SYMPTOMS.values())] == 0).all()
    assert df.loc[2, list(brazil.SYMPTOMS.values())].isna().all()
    assert df["male"].tolist() == [1, 0, 1]
    assert df["age_60"].tolist() == [0, 1, 0]
    assert "contact" not in df.columns


def test_brazil_other_test_types(tmp_path):
    df = brazil.load(_brazil_dir(tmp_path / "br"), test_types=("3",))
    assert len(df) == 1 and df["y"].iloc[0] == 1


def test_brazil_read_raw(tmp_path):
    raw = brazil.read_raw(_brazil_dir(tmp_path / "br"))
    assert len(raw) == 5 and (raw["file_year"] == 2021).all()
    assert not brazil.available(tmp_path / "empty")
