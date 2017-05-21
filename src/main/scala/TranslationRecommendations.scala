import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorIndexer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.SparkSession

import scala.util.Random

object TranslationRecommendations {
  case class Features(sitelinkCount: Int)

  def getFeatures(wikidataId: Int): Features = {
    Features(Random.nextInt(300))
  }

  def getFeatureVector(features: Features): Array[Double] = {
    Array(getPredictedRank(features), features.sitelinkCount.toDouble)
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

//    val wikidataIds = Array("Q89", "Q90", "Q91", "Q92")
    val wikidataIds = Array.fill(1000)(Random.nextInt)
    val ids = spark.sparkContext.parallelize(wikidataIds)
    val features = ids.map(getFeatures)
    val featureVectors = features.map(getFeatureVector)

    val labeled = featureVectors.map(vector => LabeledPoint(vector(0), new DenseVector(vector.drop(1))))

    val dataFrame = spark.createDataFrame(labeled).toDF("label", "features")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(dataFrame)

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val pipeline = new Pipeline().setStages(Array(featureIndexer, rf))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(500)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
//    println("Learned regression forest model:\n" + rfModel.toDebugString)

    val collected = labeled.collect()
//    println(collected.deep.mkString("\n"))
    spark.stop()
  }
}
