package org.wikimedia.research.recommendation.job.translation

import java.io.File

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import scala.collection.parallel.mutable.ParArray

object ScorePredictor {
  val log: Logger = LogManager.getLogger(ScorePredictor.getClass)

  def predictScores(spark: SparkSession,
                    modelsInputDir: File,
                    predictionsOutputDir: Option[File],
                    sites: ParArray[String],
                    featureData: DataFrame): Unit = {
    log.info("Scoring items")
    sites.foreach(target => {
      try {
        log.info("Scoring for " + target)
        log.info("Getting work data for " + target)
        val workData: DataFrame = Utils.getWorkData(spark, featureData, target, exists = false)
        log.info("Loading model for " + target)
        val model = RandomForestRegressionModel.load(
          new File(modelsInputDir, target).getAbsolutePath)
        log.info("Scoring data for " + target)
        val predictions = model.transform(workData).select("id", Utils.PREDICTION)

        predictions.show(5)
        log.info("Saving scores for " + target)
        predictionsOutputDir.foreach(o =>
          predictions.write.mode(SaveMode.ErrorIfExists).csv(new File(o, target).getAbsolutePath))
      } catch {
        case unknown: Throwable => log.error("Score for " + target + " failed", unknown)
      }
    })
  }
}
