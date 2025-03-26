import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

def get_full_cross_join(cross_join_df: DataFrame) -> DataFrame:
    target_df = cross_join_df.withColumns({
        "SourceId": F.col("TargetId"),
        "TargetId": F.col("SourceId"),
    })

    result_df = cross_join_df.unionByName(target_df)

    return result_df


def calculate_deltas(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "dlat",
        (F.col("target.latitude") - F.col("source.latitude"))
        * F.lit(math.pi / 180.0),
    ).withColumn(
        "dlon",
        (F.col("target.longitude") - F.col("source.longitude"))
        * F.lit(math.pi / 180.0),
    )


def calculate_haversine_formula(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "a",
        F.pow(F.sin(F.col("dlat") / 2), 2)
        + F.cos(F.col("source.latitude") * F.lit(math.pi / 180.0))
        * F.cos(F.col("target.latitude") * F.lit(math.pi / 180.0))
        * F.pow(F.sin(F.col("dlon") / 2), 2),
    ).withColumn("c", 2 * F.atan2(F.sqrt(F.col("a")), F.sqrt(1 - F.col("a"))))


def calculate_haversine_distance(
    df: DataFrame, R: float = 6371.0
) -> DataFrame:
    return df.withColumn("haversine_distance", F.lit(R) * F.col("c"))


def normalize_haversine_distance(df: DataFrame) -> DataFrame:
    window_spec = Window.partitionBy("SourceId")
    return df.withColumn(
        "sim_haversine",
        F.when(
            F.max("haversine_distance").over(window_spec) == 0,
            F.lit(1.0),
        ).otherwise(
            1
            - (
                F.col("haversine_distance")
                / F.max("haversine_distance").over(window_spec)
            )
        ),
    )


def calculate_haversine_similarity(df: DataFrame) -> DataFrame:
    deltas_df = calculate_deltas(df)
    haversine_df = calculate_haversine_formula(deltas_df)
    distance_df = calculate_haversine_distance(haversine_df).select(
        F.col("source.id").alias("SourceId"),
        F.col("target.id").alias("TargetId"),
        "haversine_distance",
    )

    full_cross_join_df = get_full_cross_join(distance_df)

    normalized_df = normalize_haversine_distance(full_cross_join_df)

    return normalized_df
