from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feast_repo")

entity_df = pd.read_parquet(
    "feast_repo/data/train_sample.parquet"
)

# Use a sample of 7000 rows for PIT retrieval
entity_df = entity_df.sample(n=7000, random_state=42)

entity_df["event_timestamp"] = pd.to_datetime(entity_df["timestamp"])

training_df = store.get_historical_features(
    entity_df=entity_df[
        ["stock_name", "event_timestamp"]
    ],
    features=[
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
    ],
).to_df()

print(training_df.head())

training_df.to_csv(
    "feast_repo/data/training_features.csv",
    index=False,
)

print(f"Retrieved {len(training_df)} rows from Feast")
print("Saved training_features.csv")