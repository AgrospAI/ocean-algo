from logging import getLogger
from typing import Mapping, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

logger = getLogger(__name__)


class Imputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        imputers: Optional[Mapping[str, SimpleImputer]] = None,
        skewness_threshold: float = 0.5,
    ):
        self.imputers = imputers if imputers else {}
        self.skewness_threshold = skewness_threshold

    def fit(self, X, y=None):
        # Analyze the columns and fill the missing values with the proper strategy
        skewness = X.skew(axis=1, skipna=True).abs()
        for col in X.columns:
            column_type = X[col].dtype

            # If value is categorical, fill with most frequent value (mode)
            if column_type == "object" or column_type == "category":
                imputer = SimpleImputer(strategy="most_frequent")
            elif skewness[col] < self.skewness_threshold:
                # If the column is normally distributed, fill with mean
                imputer = SimpleImputer(strategy="mean")
            else:
                # If the column is skewed, fill with median
                imputer = SimpleImputer(strategy="median")

            logger.info(f"Fitting `{imputer.strategy}` imputer for column {col}")
            imputer.fit(X[col].values.reshape(-1, 1))
            self.imputers[col] = imputer

        return self

    def transform(self, X):
        X = X.copy()
        for col, imputer in self.imputers.items():
            X[col] = imputer.transform(X[col].values.reshape(-1, 1)).ravel()
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features
