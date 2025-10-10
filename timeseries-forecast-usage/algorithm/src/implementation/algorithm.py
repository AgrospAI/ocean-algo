import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
from ocean_runner import Algorithm

from .data import InputParameters

# === Constants ===
BASE_ALGORITHM = Path("/algorithm") / "data"
MODEL = BASE_ALGORITHM / "model.pkl"
# =================


def validate(_):
    assert MODEL.exists() and MODEL.is_file()


def forecast_next_days(
    df,
    preprocessor,
    model,
    target_col,
    datetime_col,
    steps,
    lag_diff,
    lag_type,
    logger,
):
    """
    Generate recursive forecasts for the next N days using the trained preprocessor + model.
    """

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    for idx in range(steps + 1):
        X = preprocessor.transform(df)
        y_pred = model.predict(X[-1:])[0]

        # Compute next date
        next_date = df[datetime_col].iloc[-1] + timedelta(**{f"{lag_type}": lag_diff})

        # Append prediction as next day's sales
        df = pd.concat(
            [df, pd.DataFrame({datetime_col: [next_date], target_col: [y_pred]})],
            ignore_index=True,
        )

        logger.info(f"Predicted {idx}/{lag_diff}")

    return df.tail(steps + 2).set_index(datetime_col)


def run(algorithm: Algorithm) -> pd.DataFrame:

    params: InputParameters = algorithm.job_details.input_parameters

    # Load the forecasting transformer and model
    with open(MODEL, "rb") as f:
        pipeline = pickle.load(f)

    # Load the prediction data
    _, data_path = next(algorithm.job_details.next_path())
    data = pd.read_csv(
        data_path,
        index_col=0,
        sep=params.separator,
        compression=("zip" if params.is_zipped else "infer"),
    )

    return forecast_next_days(
        df=data,
        preprocessor=pipeline[:-1],
        model=pipeline.named_steps["model"],
        target_col="Sales",
        datetime_col="Date",
        steps=params.predict_steps,
        lag_diff=params.lag_diff,
        lag_type=params.lag_type,
        logger=algorithm.logger,
    )


def save_data(results: pd.DataFrame, base_path: Path, **kwargs):
    results.to_csv(base_path / "predictions.csv")
