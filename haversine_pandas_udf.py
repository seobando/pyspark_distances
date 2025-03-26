import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from sklearn.metrics.pairwise import haversine_distances

from feature_store.transformations.utils import get_full_cross_join


@pandas_udf(DoubleType())
def haversine_udf(
    src_lat: pd.Series,
    src_lon: pd.Series,
    tgt_lat: pd.Series,
    tgt_lon: pd.Series,
) -> pd.Series:
    # Convert degrees to radians
    src_coords = np.radians(np.vstack([src_lat, src_lon]).T)
    tgt_coords = np.radians(np.vstack([tgt_lat, tgt_lon]).T)

    # Compute haversine distances (returns in radians)
    distances = (
        haversine_distances(src_coords, tgt_coords) * 6371
    )  # Earth radius in km

    return pd.Series(distances.diagonal())  # Extract the diagonal (pairwise)


def normalize_haversine_distance(df: DataFrame) -> DataFrame:
    window_spec = Window.partitionBy("SourceId")
    return df.withColumn(
        "max_distance", F.max("haversine_distance").over(window_spec)
    ).withColumn(
        "sim_haversine",
        F.when(
            F.col("max_distance") == 0,
            F.lit(1.0),
        ).otherwise(1 - (F.col("haversine_distance") / F.col("max_distance"))),
    )


def calculate_haversine_similarity(df: DataFrame) -> DataFrame:
    # Calculate haversine distance using pandas UDF
    haversine_distance_df = df.withColumn(
        "haversine_distance",
        haversine_udf(
            F.col("source.latitude"),
            F.col("source.longitude"),
            F.col("target.latitude"),
            F.col("target.longitude"),
        ),
    ).select(
        F.col("source.id").alias("SourceId"),
        F.col("target.id").alias("TargetId"),
        "haversine_distance",
    )
    full_cross_join_df = get_full_cross_join(haversine_distance_df)
    # Normalize the distance to similarity score with specific partition columns
    normalized_result_df = normalize_haversine_distance(full_cross_join_df)
    return normalized_result_df
