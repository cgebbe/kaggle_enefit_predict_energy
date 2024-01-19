# %%

from enefitscripts.s import data, _ipython

_ipython.setup_autoreload()


# %%


from pathlib import Path
import pandas as pd
import polars as pl
from IPython.display import display


dirpath = data.get_input_dirpath()
for f in sorted(dirpath.glob("*.csv")):
    print(f)
    df = pl.read_csv(f)
    display(df.head(3))
    display(df.describe())


# %%
t0 = pl.read_csv(dirpath / "train.csv")
t0.describe()


# %%


def convert_datetime_polars(df: pl.DataFrame):
    return df.with_columns(
        pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"),
    ).with_columns(
        date=pl.col("datetime").dt.date(),
        hour=pl.col("datetime").dt.hour(),
        weekday=pl.col("datetime").dt.weekday(),
        month=pl.col("datetime").dt.month(),
        year=pl.col("datetime").dt.year(),
    )


def convert_datetime_pandas(df: pd.DataFrame):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    return df


t1 = convert_datetime_polars(t0)
t1.head(3)

# %%

curr_date = t1["date"].max()
test = t1.filter(pl.col("date") == curr_date)
train = t1.filter(pl.col("date") < curr_date)

print(test.shape)
print(train.shape)
print(t1.shape)

# %%

import dataclasses
import numpy as np


@dataclasses.dataclass
class Xy:
    X: np.ndarray
    y: np.ndarray
    row_id: np.ndarray


def to_Xy(
    df: pl.DataFrame,
    exclude_cols=["datetime", "data_block_id", "row_id"],
    target_col="target",
):
    return Xy(
        X=df.select(pl.exclude(target_col, *exclude_cols)).to_numpy(),
        y=df.select(target_col).to_numpy().squeeze(axis=1),
        row_id=df.select("row_id").to_numpy().squeeze(axis=1),
    )


train_xy = to_Xy(train)
test_xy = to_Xy(test)
test_xy

# %%

import lightgbm as lgb

train_data = lgb.Dataset(train_xy.X, train_xy.y)
params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}
bst = lgb.train(params, train_data, num_boost_round=100)
# %%

y_pred = bst.predict(test_xy.X, num_iteration=bst.best_iteration)
y_pred

# %%
y_pred_baseline = np.full_like(test_xy.y, fill_value=t1["target"].mean())

# %%
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)

mean_absolute_error(test_xy.y, y_pred)
# 179 for model
# 447 for baseline
