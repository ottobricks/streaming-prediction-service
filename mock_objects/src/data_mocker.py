"""DataMocker module: generate mock data for streaming-prediction-service"""

from pyspark.mllib.random import RandomRDDs
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import lit, row_number

from utils.data_mocker_argparser import parse_args


def generate_mock_binary_featureframe(
    *,
    output_path: str,
    nrows: int,
    nfeatures_poisson: int,
    nfeatures_normal: int,
    nfeatures_poisson_categorical: int,
    negative_class_proportion: float,
) -> int:
    """Public interface to generate binary-perdiction-target data"""
    spark = SparkSession.builder.getOrCreate()

    poisson_features: DataFrame = (
        RandomRDDs.poissonVectorRDD(
            spark.sparkContext, mean=5, numRows=nrows, numCols=nfeatures_poisson
        )
        .map(lambda row: [int(item) for item in row])
        .toDF([f"poisson_feature_{idx}" for idx in range(nfeatures_poisson)])
        .withColumn("id", row_number().over(Window.orderBy(lit(1))))
    )

    normal_features: DataFrame = (
        RandomRDDs.normalVectorRDD(spark.sparkContext, numRows=nrows, numCols=nfeatures_normal)
        .map(lambda row: [float(item) for item in row])
        .toDF([f"normal_feature_{idx}" for idx in range(nfeatures_normal)])
        .withColumn("id", row_number().over(Window.orderBy(lit(1))))
    )

    categorical_features: DataFrame = (
        RandomRDDs.poissonVectorRDD(
            spark.sparkContext, mean=2, numRows=nrows, numCols=nfeatures_poisson_categorical
        )
        .map(lambda row: [chr(ord("@") + int(item)) for item in row])
        .toDF(
            [
                f"categorical_feature_{idx}"
                for idx in range(nfeatures_poisson_categorical)
            ]
        )
        .withColumn("id", row_number().over(Window.orderBy(lit(1))))
    )

    label_frame: DataFrame = (
        RandomRDDs.uniformRDD(spark.sparkContext, nrows, seed=42)
        .map(lambda item: [int(item > negative_class_proportion)])
        .toDF(["label"])
        .withColumn("id", row_number().over(Window.orderBy(lit(1))))
    )

    feature_frame: DataFrame = (
        poisson_features.join(normal_features, on=["id"])
        .join(categorical_features, on=["id"])
        .join(label_frame, on=["id"])
    )

    feature_frame.write.mode("overwrite").parquet(output_path)

    return 0


if __name__ == "__main__":

    prediction_target_generator_map = {"binary": generate_mock_binary_featureframe}

    args = parse_args()
    prediction_target_type = args.pop("prediction_target_type")
    prediction_target_generator_map[prediction_target_type](**args)
