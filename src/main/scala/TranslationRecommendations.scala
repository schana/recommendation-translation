import org.apache.spark.ml.feature.LabeledPoint

import scala.util.Random
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg._

object TranslationRecommendations {
  case class Features(sitelinkCount: Int)

  def getFeatures(wikidataId: String): Features = {
    Features(Random.nextInt(300))
  }

  def getFeatureVector(features: Features): Array[Double] = {
    Array(getPredictedRank(features), features.sitelinkCount.toDouble)
  }

  def getPredictedRank(features: Features): Double = {
    features.sitelinkCount.toDouble / 300.0
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RecommendationTranslationFeatures").setMaster("local")
    val sc = new SparkContext(conf)

    val wikidataIds = sc.parallelize(Array("Q89", "Q90", "Q91", "Q92"))
    val features = wikidataIds.map(getFeatures)
    val featureVectors = features.map(getFeatureVector)

    val labeled = featureVectors.map(vector => LabeledPoint(vector(0), new DenseVector(vector.drop(1))))

    val collected = labeled.collect()
    println(collected.deep.mkString("\n"))
    sc.stop()
  }
}
