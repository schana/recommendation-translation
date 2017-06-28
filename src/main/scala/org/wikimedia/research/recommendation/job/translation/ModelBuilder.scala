package org.wikimedia.research.recommendation.job.translation

import java.io.File

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.parallel.mutable.ParArray

object ModelBuilder {
  val log: Logger = LogManager.getLogger(ModelBuilder.getClass)

  def buildModels(spark: SparkSession,
                  modelsOutputDir: Option[File],
                  sites: ParArray[String],
                  featureData: DataFrame): Unit = {
    log.info("Building Models")
    sites.foreach(target =>
      try {
        log.info("Building model for " + target)
        log.info("Getting work data for " + target)
        val workData: DataFrame = Utils.getWorkData(spark, featureData, target)
        val Array(trainingData, testData) = workData.randomSplit(Array(0.7, 0.3))
        log.info("Training model for " + target)
        val model = Utils.REGRESSOR.fit(trainingData)
        log.info("Writing model to file for " + target)
        modelsOutputDir.foreach(o => model.write.save(new File(o, target).getAbsolutePath))
        log.info("Testing model for " + target)
        val predictions = model.transform(testData)
        predictions.show(5)
        val rmse = Utils.EVALUATOR.evaluate(predictions)
        log.info("Root Mean Squared Error (RMSE) on test data for " + target + " = " + rmse)
      } catch {
        case unknown: Throwable => log.error("Build model for " + target + " failed", unknown)
      }
    )
  }
}
