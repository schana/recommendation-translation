package org.wikimedia.research.recommendation.job.translation

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.{DataFrame, SparkSession}

case class SitelinkEntry(id: String, site: String, title: String)

case class PagecountEntry(site: String, title: String, pageviews: Double)

case class SitelinkPageviewsEntry(id: String, site: String, title: String, pageviews: Double)

case class RankedEntry(id: String, site: String, title: String, pageviews: Double, rank: Double)

object Utils {
  val FEATURES = "features"
  val LABEL = "label"
  val PREDICTION = "prediction"
  val EXISTS = 1.0
  val NOT_EXISTS = 0.0
  val REGRESSOR: RandomForestRegressor = new RandomForestRegressor()
    .setLabelCol(LABEL)
    .setFeaturesCol(FEATURES)
  val EVALUATOR: RegressionEvaluator = new RegressionEvaluator()
    .setLabelCol(LABEL)
    .setPredictionCol(PREDICTION)
    .setMetricName("rmse")

  def getWorkData(spark: SparkSession, data: DataFrame, target: String, exists: Boolean = true): DataFrame = {
    val workData: DataFrame = data.filter(row =>
      row(row.fieldIndex("exists_" + target)) == (if (exists) EXISTS else NOT_EXISTS))

    import spark.implicits._
    val labeledData = workData.map(row =>
      (
        row.getString(row.fieldIndex("id")),
        row.getDouble(row.fieldIndex("rank_" + target)),
        /* Include everything except the target language's features */
        new DenseVector((
          (1 until row.fieldIndex("pageviews_" + target)).map(row.getDouble) ++
            (row.fieldIndex("exists_" + target) + 1 until row.length).map(row.getDouble)
          ).toArray)
      )
    ).rdd

    spark.createDataFrame(labeledData).toDF("id", LABEL, FEATURES)
  }
}
