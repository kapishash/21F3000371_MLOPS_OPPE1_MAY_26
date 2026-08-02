import mlflow
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


model = mlflow.pyfunc.load_model(
    "models:/m-88410b189d6f44d1ba7cf17e82d4b3ce"
)


encoder = joblib.load(
    "models/label_encoder.pkl"
)


test = pd.read_parquet(
    "feast_repo/data/merged/test.parquet"
)


# same preprocessing as training
test["stock_name"] = encoder.transform(
    test["stock_name"]
)


features = [
    "rolling_avg_10",
    "volume_sum_10",
    "stock_name"
]


X_test = test[features]

y_test = test["target"]


pred = model.predict(X_test)


print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

print(
    "F1:",
    f1_score(y_test, pred)
)


# Feature sanity tests

assert "rolling_avg_10" in test.columns
assert "volume_sum_10" in test.columns
assert "stock_name" in test.columns

assert test["rolling_avg_10"].notnull().all()
assert test["volume_sum_10"].notnull().all()


print(
    "Feature sanity tests passed"
)