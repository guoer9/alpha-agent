"""Alpha Agent 特征定义"""
from datetime import timedelta
from feast import Entity, Feature, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# 定义股票实体
stock = Entity(
    name="stock",
    join_keys=["symbol"],
    value_type=String,
    description="股票代码",
)

# 定义因子特征源
factor_source = FileSource(
    path="data/factors.parquet",
    timestamp_field="date",
)

# 定义因子特征视图
factor_features = FeatureView(
    name="factor_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="momentum", dtype=Float32),
        Field(name="volatility", dtype=Float32),
        Field(name="volume_ratio", dtype=Float32),
        Field(name="rsi", dtype=Float32),
        Field(name="macd", dtype=Float32),
    ],
    source=factor_source,
)