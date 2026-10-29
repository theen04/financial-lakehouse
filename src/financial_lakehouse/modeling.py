import logging
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import GOLD_BASELINE_METRICS
from .datasets import TREND_LABEL
from .gold import FEATURE_COLUMNS
from .storage import write_table


logger = logging.getLogger(__name__)

SEED = 42

BASELINE_LABELS = ["direction_5d", TREND_LABEL]


def _models(label: str) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            featuresCol="features",
            labelCol=label,
            maxIter=100,
            regParam=0.01,
        ),
        "random_forest": RandomForestClassifier(
            featuresCol="features",
            labelCol=label,
            numTrees=100,
            maxDepth=6,
            seed=SEED,
        ),
    }


def _majority_accuracy(train: DataFrame, test: DataFrame, label: str) -> float:
    """Accuracy of always predicting the most common training class."""

    majority = (
        train.groupBy(label).count()
        .orderBy(F.col("count").desc())
        .first()[label]
    )

    return test.select(
        F.avg((F.col(label) == majority).cast("double"))
    ).first()[0]


def evaluate_baselines(
    spark: SparkSession,
    classification_df: DataFrame,
    labels: list[str] = BASELINE_LABELS,
) -> DataFrame:
    """
    Train baseline classifiers on the train split and score validation and test.

    This checks that the classification dataset is usable end to end and
    sets a reference point against a majority-class baseline; it is not
    a tuned trading model.
    """
    df = classification_df.cache()
    train = df.filter("split = 'train'")

    assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features")

    rows = []

    for label in labels:
        accuracy = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")
        weighted_f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")

        for name, model in _models(label).items():
            logger.info("Training %s for %s", name, label)

            fitted = Pipeline(stages=[assembler, scaler, model]).fit(train)

            for split in ("validation", "test"):
                scored_split = df.filter(F.col("split") == split)
                predictions = fitted.transform(scored_split)

                rows.append({
                    "label": label,
                    "model": name,
                    "split": split,
                    "accuracy": accuracy.evaluate(predictions),
                    "f1_weighted": weighted_f1.evaluate(predictions),
                    "majority_baseline_accuracy": _majority_accuracy(
                        train, scored_split, label
                    ),
                })

    df.unpersist()

    return spark.createDataFrame(rows)


def write_baseline_metrics(
    df: DataFrame,
    output_dir: Path = GOLD_BASELINE_METRICS,
) -> None:
    """Write baseline metrics to Delta."""

    write_table(df, output_dir)
