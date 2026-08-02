import os
import argparse
import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--iteration",
    default="v0",
    choices=["v0", "merged"]
)

args = parser.parse_args()


# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("stock_movement_prediction")


# Select dataset based on iteration
if args.iteration == "v0":
    data_path = "feast_repo/data/v0"

else:
    data_path = "feast_repo/data/merged"


# Load data
train = pd.read_parquet(
    os.path.join(data_path, "train.parquet")
)

test = pd.read_parquet(
    os.path.join(data_path, "test.parquet")
)


# Encode stock name
encoder = LabelEncoder()

train["stock_name"] = encoder.fit_transform(
    train["stock_name"]
)

test["stock_name"] = encoder.transform(
    test["stock_name"]
)


features = [
    "rolling_avg_10",
    "volume_sum_10",
    "stock_name",
]


X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


# Hyperparameter sweep
configs = [
    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6
    },
    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 8
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    }
]


best_f1 = -1
best_model = None


for config in configs:

    with mlflow.start_run():

        mlflow.set_tag(
            "iteration",
            args.iteration
        )

        mlflow.set_tag(
            "model",
            "LightGBM"
        )


        model = LGBMClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=42
        )


        model.fit(
            X_train,
            y_train
        )


        pred = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            pred
        )

        precision = precision_score(
            y_test,
            pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            pred,
            zero_division=0
        )


        mlflow.log_params(
            config
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        )


        mlflow.lightgbm.log_model(
            model,
            "model"
        )


        print(
            f"{config} F1={f1:.4f}"
        )


        if f1 > best_f1:
            best_f1 = f1
            best_model = model



# Save best model locally
os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/model.pkl"
)


joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)


# Register best model in MLflow
with mlflow.start_run(run_name="best_model_registration"):

    mlflow.set_tag(
        "iteration",
        args.iteration
    )

    mlflow.log_metric(
        "best_f1",
        best_f1
    )

    model_info = mlflow.lightgbm.log_model(
        best_model,
        name="model"
    )

    print(
        "Registered model URI:",
        model_info.model_uri
    )



print("\nBest model saved")
print("Iteration:", args.iteration)
print("Best F1:", best_f1)