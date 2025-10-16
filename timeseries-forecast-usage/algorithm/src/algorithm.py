import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
from ocean_runner import Algorithm, Config

from .data import InputParameters

# === Constants ============================
BASE_ALGORITHM = Path("/algorithm") / "data"
MODEL = BASE_ALGORITHM / "model.pkl"
# ==========================================

algorithm = Algorithm(config=Config(custom_input=InputParameters))


@algorithm.validate
def validate(*args, **kwargs):
    assert MODEL.exists() and MODEL.is_file()


@algorithm.run
def run(algorithm: Algorithm) -> pd.DataFrame:
    params: InputParameters = algorithm.job_details.input_parameters

    # Load the forecasting transformer and model
    with open(MODEL, "rb") as f:
        pipeline = pickle.load(f)

    preprocessor = pipeline[:-1]
    model = pipeline.named_steps["model"]

    # Load the prediction data
    _, data_path = next(algorithm.job_details.next_path())
    df = pd.read_csv(
        data_path,
        index_col=0,
        sep=params.separator,
        compression=("zip" if params.is_zipped else "infer"),
    )

    df[params.datetime_col] = pd.to_datetime(df[params.datetime_col])
    df = df.sort_values(params.datetime_col)

    for idx in range(params.predict_steps + 1):
        X = preprocessor.transform(df)
        y_pred = model.predict(X[-1:])[0]

        # Compute next date
        next_date = df[params.datetime_col].iloc[-1] + timedelta(
            **{f"{params.lag_type}": params.lag_diff}
        )

        # Append prediction as next day's sales
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {params.datetime_col: [next_date], params.target_col: [y_pred]}
                ),
            ],
            ignore_index=True,
        )

        algorithm.logger.info(f"Predicted {idx}/{params.lag_diff}")

    return df.tail(params.predict_steps + 2).set_index(params.datetime_col)


@algorithm.save_results
def save(result: pd.DataFrame, base: Path, *args, **kwargs):
    result.to_csv(base / "predictions.csv")


if __name__ == "__main__":
    algorithm()
