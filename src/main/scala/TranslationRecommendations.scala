import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.SparkSession

import scala.util.Random

object TranslationRecommendations {
  case class Features(sitelinkCount: Int, recentPageviews: Int)

  def getFeatures(wikidataId: Int): Features = {
    Features(Random.nextInt(300), Random.nextInt(10000))
  }

  def getLabeled(rankedFeature: (Features, Long)): LabeledPoint = {
    val (features, rank) = rankedFeature
    LabeledPoint(rank.toDouble, getFeatureVector(features))
  }

  def getFeatureVector(features: Features): DenseVector = {
    new DenseVector(Array(
      features.sitelinkCount.toDouble,
      features.recentPageviews.toDouble
    ))
  }

  def getPredictedRank(features: Features): Double = {
    features.sitelinkCount.toDouble / 300.0
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("TranslationRecommendations")
      .master("local")
      .getOrCreate()

    // Get features for a list of wikidata ids
    val wikidataIds = (80 to 500).toArray
    val ids = spark.sparkContext.parallelize(wikidataIds)
    val featuresList = ids.map(getFeatures)

    // Get the rank based on sorting by pageviews
    val sortedFeatures = featuresList.sortBy(features => features.recentPageviews, ascending = false)
    val rankedFeatures = sortedFeatures.zipWithIndex()

    // Massage the data to be in a format used by machine learning library
    val labeled = rankedFeatures.map(getLabeled)
    val dataFrame = spark.createDataFrame(labeled).toDF("label", "features")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(dataFrame)

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    val regressor = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

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

//    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
//    println("Learned regression forest model:\n" + rfModel.toDebugString)

//    val collected = labeled.collect()
//    println(collected.deep.mkString("\n"))
    spark.stop()
  }
}
