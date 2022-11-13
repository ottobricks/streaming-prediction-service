"""ModelMocker module: generate mock ML model for streaming-prediction-service"""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import coalesce, col, lit
from pyspark.sql.utils import AnalysisException
from xgboost.spark import SparkXGBClassifier


def _create_spark_session() -> SparkSession:
    return SparkSession.builder.config(
        "spark.jars.packages",
        "org.apache.hadoop:hadoop-aws:3.2.0,"
        + "org.apache.spark:spark-hadoop-cloud_2.12:3.3.1,"
        + "ml.dmlc:xgboost4j-spark_2.12:1.7.1",
    ).getOrCreate()


def _load_training_data(spark: SparkSession, input_path: str) -> DataFrame:
    """Load data to fit a mock ML model"""
    try:
        return spark.read.parquet(input_path)
    except AnalysisException as excpt:
        raise Exception(
            f"ModelMocker: exception found when loading training data from path: {input_path}"
        ) from excpt


def _split_train_validation_data(train_df: DataFrame, valid_percent: float) -> DataFrame:
    """Add column to tag rows as train or validation via stratified sampling"""
    assert valid_percent > 0.0 and valid_percent < 1.0

    valid_df = train_df.sampleBy(
        "label", fractions={0: valid_percent, 1: valid_percent}
    )

    return train_df.join(
        valid_df.select("id", lit(True).alias("is_validation")), on=["id"], how="left"
    ).withColumn("is_validation", coalesce(col("is_validation"), lit(False)))


def _fit_pipeline_model(train_df: DataFrame) -> PipelineModel:
    """Fit a PipelineModel containing at least one XGBoost estimator"""

    ohe_transformer = OneHotEncoder(
        inputCols=[
            column for column in train_df.columns if column.startswith("categorical")
        ],
        outputCols=[
            f"{column}_ohe"
            for column in train_df.columns
            if column.startswith("categorical")
        ],
        handleInvalid="keep",
    )

    vec_assembler = VectorAssembler(
        inputCols=[
            f"{column}_ohe" if column.startswith("categorical") else column
            for column in train_df.columns
        ],
        outputCol="features",
        handleInvalid="keep",
    )

    xgb_classifier = SparkXGBClassifier(
        max_depth=10,
        missing=0.0,
        validation_indicator_col="is_validation",
        weight_col="weight",
        early_stopping_rounds=1,
        eval_metric="aucpr",
        num_workers=-1,
    )

    return Pipeline(stages=[ohe_transformer, vec_assembler, xgb_classifier])
