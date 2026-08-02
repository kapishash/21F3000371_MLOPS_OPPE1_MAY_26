from datetime import timedelta

from feast import FeatureView, FileSource, Field
from feast.types import Float32

from entities import stock

stock_source = FileSource(
    path="data/train_sample.parquet",
    timestamp_field="timestamp",
)

stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_source,
    online=True,
)