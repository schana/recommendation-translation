import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

case class SitelinkPageviewsEntry(wikidataId: String, site: String, title: String, pageviews: Int)
case class AggregateSitelinkPageviews(wikidataId: String, sitelinkCount: Int, pageviews: Int)
case class Features(wikidataId: String, sitelinkCount: Int, recentPageviews: Int)

object TranslationRecommendations {
  val FEATURES = "features"
  val INDEXED_FEATURES = "indexedFeatures"
  val LABEL = "label"
  val PREDICTION = "prediction"

  def main(args: Array[String]): Unit = {
    val spark = initializeSpark()

    val sitelinkPageviewsEntries = getSitelinkPageviewsEntries(spark).persist()
    val data = transformEntriesToFeatures(spark, sitelinkPageviewsEntries)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val regressor = new RandomForestRegressor()
      .setLabelCol(LABEL)
      .setFeaturesCol(FEATURES)
      .setNumTrees(50)

    val pipeline = new Pipeline().setStages(Array(regressor))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    predictions.show(500)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(LABEL)
      .setPredictionCol(PREDICTION)
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    spark.stop()
  }

  def initializeSpark(): SparkSession = {
    SparkSession
      .builder()
      .appName("TranslationRecommendations")
      .master("local")
      .getOrCreate()
  }

  def getSitelinkPageviewsEntries(spark: SparkSession): RDD[SitelinkPageviewsEntry] = {
    spark.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .csv("lite-sitelink-pageviews.csv")
      .rdd
      .map(row => SitelinkPageviewsEntry(row.getString(1), row.getString(2), row.getString(3), row.getString(4).toInt))
  }

  def transformEntriesToFeatures(spark: SparkSession, entries: RDD[SitelinkPageviewsEntry]): DataFrame = {
    val preReduce = entries.map(sitelinkPageviewsEntry => AggregateSitelinkPageviews(
      sitelinkPageviewsEntry.wikidataId, 1, sitelinkPageviewsEntry.pageviews))

    val grouped = preReduce.keyBy(item => item.wikidataId)
    val aggregateSitelinkPageviews = grouped.reduceByKey((a, b) =>
      AggregateSitelinkPageviews(a.wikidataId, a.sitelinkCount + b.sitelinkCount, a.pageviews + b.pageviews))
    val features = aggregateSitelinkPageviews.map(item => Features(item._1, item._2.sitelinkCount, item._2.pageviews))

    val rankedFeatures = rankFeatures(features)

    prepareDataFrame(spark, rankedFeatures)
  }

  def rankFeatures(features: RDD[Features]): RDD[(Features, Double)] = {
    // Sort by pageviews
    val sortedFeatures = features.sortBy(featuresInstance => featuresInstance.recentPageviews, ascending = false)
    val rankedFeatures = sortedFeatures.zipWithIndex()
    // Normalize rank
    val count = rankedFeatures.count()
    rankedFeatures.map(f => normalizeRankedFeatures(f, count))
  }

  def normalizeRankedFeatures(rankedFeatures: (Features, Long), rankCount: Long): (Features, Double) = {
    val (features, rank) = rankedFeatures
    (features, rank.toDouble / rankCount.toDouble)
  }

  def prepareDataFrame(spark: SparkSession, rankedFeatures: RDD[(Features, Double)]): DataFrame = {
    val labeled = rankedFeatures.map{case (features, rank) => (features.wikidataId, rank, getFeatureVector(features))}
    spark.createDataFrame(labeled).toDF("id", LABEL, FEATURES)
  }

  def getFeatureVector(features: Features): DenseVector = {
    new DenseVector(Array(
      features.sitelinkCount.toDouble,
      features.recentPageviews.toDouble
    ))
  }
}
