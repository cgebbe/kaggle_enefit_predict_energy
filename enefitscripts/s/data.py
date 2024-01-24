import socket
from pathlib import Path
import polars as pl
import pandas as pd
import dataclasses
import numpy as np


def is_local():
    return socket.gethostname() == "cgebbe"


def get_input_dirpath():
    s = {
        True: "/mnt/sda1/projects/git/competitions/20240118_kaggle_enefit",
        False: "/kaggle/input",
    }[is_local()]
    p = Path(s)
    assert p.exists(), p
    return p


def get_data_dirpath():
    input_dirpath = get_input_dirpath()
    data_dirpath = input_dirpath / "predict-energy-behavior-of-prosumers"
    assert data_dirpath.exists()
    return data_dirpath


def _convert_datetime_polars(df: pl.DataFrame):
    col_dtype = df["datetime"].dtype
    if isinstance(col_dtype, pl.Datetime):
        pass
    # FIXME: How is this correct?!
    elif issubclass(col_dtype, pl.Utf8):
        df = df.with_columns(
            pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        raise NotImplementedError(f"{col_dtype=}")

    return df.with_columns(
        date=pl.col("datetime").dt.date(),
        hour=pl.col("datetime").dt.hour(),
        weekday=pl.col("datetime").dt.weekday(),
        month=pl.col("datetime").dt.month(),
        year=pl.col("datetime").dt.year(),
    )


def _convert_datetime_pandas(df: pd.DataFrame):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    return df


def preprocess(df: pl.DataFrame):
    t0 = _convert_datetime_polars(df)
    if "target" in t0.columns:
        # NOTE: I don't think that imputing Null values makes any sense...
        t1 = t0.filter(t0["target"].is_not_null())
    else:
        t1 = t0
    return t1


class Table:
    def __init__(self, df: pl.DataFrame) -> None:
        self.idx_cols = [
            "county",
            "is_business",
            "product_type",
            "is_consumption",
        ]
        self.df = df

    def split_train_test(self, test_date):
        return (
            Table(self.df.filter(pl.col("date") == test_date)),
            Table(self.df.filter(pl.col("date") < test_date)),
        )

    @classmethod
    def from_path(cls, path: Path):
        t0 = pl.read_csv(path)
        t1 = preprocess(t0)
        return cls(t1)

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame):
        t0 = pl.from_pandas(df).rename({"prediction_datetime": "datetime"})
        t1 = preprocess(t0)
        return cls(t1)

    def get_X(self):
        feature_cols = [
            #  'target',
            # "datetime",
            # "data_block_id",
            # "row_id",
            # "prediction_unit_id",
            # "date",
            "hour",
            "weekday",
            "month",
            "year",
        ]
        return self.df.select(self.idx_cols + feature_cols).to_numpy()

    def get_y(self):
        target_col = "target"
        return self.df.select(target_col).to_numpy().squeeze(axis=1)


import lightgbm as lgb


def load_model() -> lgb.Booster:
    model_path = get_input_dirpath() / "enefitmodel/model.txt"
    assert model_path.exists()
    return lgb.Booster(model_file=model_path)


def load_test_tables():
    data_dirpath = get_data_dirpath()
    return {
        p.stem: pd.read_csv(p) for p in data_dirpath.glob("example_test_files/*.csv")
    }


def predict(model: lgb.Booster, test_tables: dict[str, pd.DataFrame]):
    test = Table.from_pandas_df(test_tables["test"])
    y_pred = model.predict(test.get_X())

    tname = "sample_submission"
    assert all(test_tables[tname]["row_id"].to_numpy() == test.df["row_id"].to_numpy())
    test_tables[tname]["target"] = y_pred
