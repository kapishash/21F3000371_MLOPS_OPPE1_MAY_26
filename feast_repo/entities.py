from feast import Entity
from feast.value_type import ValueType

stock = Entity(
    name="stock_name",
    join_keys=["stock_name"],
    value_type=ValueType.STRING,
)