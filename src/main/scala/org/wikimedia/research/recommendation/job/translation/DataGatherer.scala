package org.wikimedia.research.recommendation.job.translation

import java.io.File

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.wikimedia.research.recommendation.job.translation.ArgParser.Params

object DataGatherer {
  val log: Logger = LogManager.getLogger(DataGatherer.getClass)

  def gatherData(spark: SparkSession, params: Params, timestamp: String): Dataset[SitelinkPageviewsEntry] = {
    val parsedData: Dataset[SitelinkPageviewsEntry] =
      if (params.parseRawData) {
        log.info("Parsing raw data")
        val p = parseRawData(spark, params.rawSitelinks, params.rawPagecounts, params.rawData)
        params.outputDir.foreach(o =>
          p.write.mode(SaveMode.ErrorIfExists).save(new File(o, timestamp + "_parsedData").getAbsolutePath))
        p
      } else {
        log.info("Reading parsed data")
        import spark.implicits._
        spark.read.load(params.parsedData.get.getAbsolutePath).as[SitelinkPageviewsEntry]
      }
    parsedData
  }

  def parseRawData(spark: SparkSession,
                   rawSitelinks: Option[File],
                   rawPagecounts: Option[File],
                   rawData: Option[File]): Dataset[SitelinkPageviewsEntry] = {
    import spark.implicits._

    if (rawData.isEmpty) {
      log.info("Reading raw sitelinks")
      val sitelinks = spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "DROPMALFORMED")
        .option("sep", "\t")
        .csv(rawSitelinks.get.getAbsolutePath)
        .as[SitelinkEntry]

      val pagecounts = getPagecounts(spark, rawPagecounts)

      sitelinks.join(pagecounts, usingColumns = Seq("site", "title"))
        .as[SitelinkPageviewsEntry]
    } else {
      log.info("Reading raw data")
      spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "DROPMALFORMED")
        .option("sep", "\t")
        .csv(rawData.get.getAbsolutePath)
        .as[SitelinkPageviewsEntry]
    }
  }

  def getPagecounts(spark: SparkSession, rawPagecounts: Option[File]): Dataset[PagecountEntry] = {
    import spark.implicits._
    if (rawPagecounts.isEmpty) {
      log.info("Reading pagecounts from hadoop")

      spark.sql(
        """
          |SELECT project as site, page_title as title, sum(view_count) as pageviews
          |FROM wmf.pageview_hourly
          |WHERE year=2017 and month=1 and day=1
          |GROUP BY project, page_title
        """.stripMargin).as[PagecountEntry]
    } else {
      log.info("Reading raw pagecounts")

      val pagecountSchema = StructType(Array(
        StructField("site", StringType),
        StructField("title", StringType),
        StructField("pageviews", DoubleType)
      ))

      val pagecounts = spark.read
        .format("csv")
        .option("quote", "\u0000")
        .option("escape", "\u0000")
        .option("mode", "DROPMALFORMED")
        .option("sep", " ")
        .schema(pagecountSchema)
        .csv(rawPagecounts.get.getAbsolutePath)
        .as[PagecountEntry]

        .filter(_.site.endsWith(".z"))
        .map(e => PagecountEntry(
          site = """\.z$""".r.replaceFirstIn(e.site, "wiki"),
          title = e.title,
          pageviews = e.pageviews))

      pagecounts
    }
  }
}
