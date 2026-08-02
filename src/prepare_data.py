import os
import glob
import pandas as pd


def process_file(file_path):
    df = pd.read_csv(file_path)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by time (important because data isn't guaranteed to be ordered)
    df = df.sort_values("timestamp")

    # Stock name
    stock_name = os.path.basename(file_path).replace("__EQ__NSE__NSE__MINUTE.csv", "")
    df["stock_name"] = stock_name

    # Handle missing values
    df = df.ffill()

    # Rolling features
    df = df.set_index("timestamp")

    df["rolling_avg_10"] = (
        df["close"]
        .rolling("10min", min_periods=1)
        .mean()
    )

    df["volume_sum_10"] = (
        df["volume"]
        .rolling("10min", min_periods=1)
        .sum()
    )

    # Target
    df["close_5min_future"] = df["close"].shift(-5)
    df["target"] = (
        df["close_5min_future"] > df["close"]
    ).astype(int)

    df = df.drop(columns=["close_5min_future"])

    df = df.reset_index()

    return df


def build_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(os.path.join(input_folder, "*.csv"))

    dfs = []

    for f in files:
        dfs.append(process_file(f))

    data = pd.concat(dfs, ignore_index=True)

    data = data.sort_values("timestamp")

    train = data.iloc[:-20]
    test = data.iloc[-20:]

    train.to_parquet(
    os.path.join(output_folder, "train.parquet"),
    index=False
    )
    
    test.to_parquet(
        os.path.join(output_folder, "test.parquet"),
        index=False
    )

    print("Train:", train.shape)
    print("Test :", test.shape)


if __name__ == "__main__":

    build_dataset(
        "data/v0",
        "feast_repo/data"
    )