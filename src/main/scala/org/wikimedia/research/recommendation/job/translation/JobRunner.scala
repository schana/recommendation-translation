package org.wikimedia.research.recommendation.job.translation

import java.io.File
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.log4j.{Level, LogManager, Logger}
import org.apache.spark.sql._

import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool

object JobRunner {
  val log: Logger = LogManager.getLogger(JobRunner.getClass)
  val SITE_PARALLELISM = 8

  def main(args: Array[String]): Unit = {
    log.info("Starting")
    val params = ArgParser.parseArgs(args)

    var sparkBuilder = SparkSession
      .builder()
      .appName("TranslationRecommendations")
    if (params.runLocally) {
      sparkBuilder = sparkBuilder.master("local[*]")
    }
    val spark = sparkBuilder.getOrCreate()
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("org.wikimedia").setLevel(Level.INFO)

    val timestamp = new SimpleDateFormat("yyyy-MM-dd-HHmmss").format(Calendar.getInstance.getTime)
    log.info("Timestamp for creating files: " + timestamp)

    /*
     * Get raw data and parse it from whatever sources are specified
     */
    val parsedData: Dataset[SitelinkPageviewsEntry] = DataGatherer.gatherData(spark, params, timestamp)

    /*
     * Extract feature vectors from parsed data
     */
    val featureData: DataFrame = FeatureExtractor.extractFeatures(spark, params, timestamp, parsedData)

    /*
     * Get list of sites to work on and enable them to be worked on in parallel
     */
    val sites = if (params.targetWikis.nonEmpty)
      params.targetWikis.toArray
    else
      parsedData.rdd.map(_.site).distinct().collect().sorted
    val modelSites = sites.par
    modelSites.tasksupport = new ForkJoinTaskSupport(
      new ForkJoinPool(SITE_PARALLELISM)
    )

    /*
     * Build the models
     */
    val modelsOutputDir = params.outputDir.map(o => new File(o, timestamp + "_models"))
    if (params.buildModels) {
      modelsOutputDir.foreach(o => o.mkdir())
      ModelBuilder.buildModels(spark, modelsOutputDir, modelSites, featureData)
    }

    /*
     * Score items using the models
     */
    if (params.scoreItems) {
      val modelsInputDir = params.modelsDir.getOrElse(modelsOutputDir.get)
      val predictionsOutputDir = params.outputDir.map(o => new File(o, timestamp + "_predictions"))
      predictionsOutputDir.foreach(o => o.mkdir())
      ScorePredictor.predictScores(spark, modelsInputDir, predictionsOutputDir, modelSites, featureData)
    }

    spark.stop()
    log.info("Finished")
  }
}
