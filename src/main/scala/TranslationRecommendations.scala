import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object TranslationRecommendations {
  case class WikidataItem(id: Int, sitelinks: Array[String])
  case class Features(sitelinkCount: Int, recentPageviews: Int)

  def main(args: Array[String]): Unit = {
    val spark = initializeSpark()

    val wikidataItems = getWikidataItems(spark)
    val features = wikidataItems.map(getFeatures)
    val rankedFeatures = rankFeatures(features)

    val data = prepareDataFrame(spark, rankedFeatures)

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val regressor = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(50)

    val pipeline = new Pipeline().setStages(Array(featureIndexer, regressor))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(500)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
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

  def getWikidataItems(spark: SparkSession): RDD[WikidataItem] = {
    val wikidataIds = (80 to 200).toArray
    val ids = spark.sparkContext.parallelize(wikidataIds)
    ids.map(id => WikidataItem(id, getSitelinks(id)))
  }

  def getSitelinks(wikidataId: Int): Array[String] = {
    Array("en", "de")
  }

  def getFeatures(wikidataItem: WikidataItem): Features = {
    Features(
      sitelinkCount=wikidataItem.sitelinks.length,
      recentPageviews=Random.nextInt(10000)
    )
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
    val labeled = rankedFeatures.map(getLabeled)
    spark.createDataFrame(labeled).toDF("label", "features")
  }

  def getLabeled(rankedFeature: (Features, Double)): LabeledPoint = {
    val (features, rank) = rankedFeature
    LabeledPoint(rank, getFeatureVector(features))
  }

  def getFeatureVector(features: Features): DenseVector = {
    new DenseVector(Array(
      features.sitelinkCount.toDouble,
      features.recentPageviews.toDouble
    ))
  }
}
