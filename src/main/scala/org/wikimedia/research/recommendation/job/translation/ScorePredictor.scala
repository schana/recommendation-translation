package org.wikimedia.research.recommendation.job.translation

import java.io.File

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.collection.parallel.mutable.ParArray

object ScorePredictor {
  val log: Logger = LogManager.getLogger(ScorePredictor.getClass)

  def predictScores(spark: SparkSession,
                    modelsInputDir: File,
                    predictionsOutputDir: Option[File],
                    sites: ParArray[String],
                    featureData: DataFrame): Unit = {
    log.info("Scoring items")

    spark.sparkContext.setCheckpointDir("/tmp")
    val predictions: Array[DataFrame] = sites.map(target => {
      try {
        log.info("Scoring for " + target)
        log.info("Getting work data for " + target)
        val workData: DataFrame = Utils.getWorkData(spark, featureData, target, exists = false)
        log.info("Loading model for " + target)
        val model = RandomForestRegressionModel.load(
          new File(modelsInputDir, target).getAbsolutePath)
        log.info("Scoring data for " + target)
        val targetPredictions = model
          .setPredictionCol(target)
          .transform(workData)
          .select("id", target)

        targetPredictions.checkpoint()
      } catch {
        case unknown: Throwable =>
          log.error("Score for " + target + " failed", unknown)
          val schema = StructType(Seq(
            StructField("id", StringType, nullable = false),
            StructField(target, DoubleType, nullable = true)))
          spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
      }
    }).toArray

    log.info("Aggregating predictions")
    val schema = StructType(Seq(
      StructField("id", StringType, nullable = false)))
    var predictedScores = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)
    predictions.foreach(prediction =>
      predictedScores = predictedScores.join(prediction, Seq("id"), "outer").checkpoint()
    )
    predictedScores.show(5)

    log.info("Saving predictions")
    predictionsOutputDir.foreach(f = o =>
      predictedScores.coalesce(1)
        .write
        .mode(SaveMode.ErrorIfExists)
        .option("header", value = true)
        .option("compression", "bzip2")
        .csv(new File(o, "allPredictions").getAbsolutePath))
  }
}
