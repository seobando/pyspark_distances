import pytest
from chispa.dataframe_comparer import assert_df_equality
from client.spark_config import spark
from pyspark.sql import Window
from pyspark.sql import functions as F

from haversine import calculate_haversine_similarity


def get_cross_join_df(df: DataFrame, id_column: str) -> DataFrame:
    return df.alias("source").join(
        df.alias("target"),
        (F.col(f"source.{id_column}") < F.col(f"target.{id_column}")),
    )



@pytest.fixture
def sample_locations_df():
    data = [
        (1, 40.7128, -74.0060, "New York"),
        (2, 34.0522, -118.2437, "Los Angeles"),
        (3, 48.8566, 2.3522, "Paris"),
        (4, 35.6895, 139.6917, "Tokyo"),
        (5, -33.8688, 151.2093, "Sydney"),
        (6, 55.7558, 37.6173, "Moscow"),
        (7, 59.9343, 30.3351, "Saint Petersburg"),
    ]
    return spark.createDataFrame(data, ["id", "latitude", "longitude", "city"])


@pytest.fixture
def expected_similar_pairs():
    data = [
        (7, 6),  # Moscow to St. Petersburg
        (3, 7),  # Paris to St. Petersburg
        (3, 6),  # Paris to Moscow
        (1, 2),  # New York to Los Angeles
        (3, 1),  # New York to Paris
    ]
    return spark.createDataFrame(data, ["SourceId", "TargetId"])


def test_haversine_similarity_pipeline(
    sample_locations_df, expected_similar_pairs
):
    input_df = sample_locations_df.select(
        F.col("id"), F.col("latitude"), F.col("longitude"), F.col("city")
    )

    cross_join_df = get_cross_join_df(input_df, "id")

    result_df = (
        calculate_haversine_similarity(cross_join_df)
        .withColumn(
            "row_num",
            F.row_number().over(
                Window.partitionBy("haversine_distance").orderBy(
                    F.desc("sim_haversine")
                )
            ),
        )
        .filter(F.col("row_num") == 1)
        .orderBy(F.desc("sim_haversine"))
        .limit(5)
    ).select("SourceId", "TargetId")

    assert_df_equality(
        expected_similar_pairs,
        result_df,
        ignore_nullable=False,
        ignore_row_order=False,
    )

    invalid_similarities = result_df.filter(
        (F.col("sim_haversine") > 1.0) | (F.col("sim_haversine") < 0.0)
    ).count()
    assert invalid_similarities == 0
