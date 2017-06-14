import java.io.File
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.Partitioner
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import scopt.OptionParser

import scala.math.Ordered.orderingToOrdered

case class SitelinkEntry(id: String, site: String, title: String)

case class PagecountEntry(site: String, title: String, pageviews: Double)

case class SitelinkPageviewsEntry(id: String, site: String, title: String, pageviews: Double)

case class RankedEntry(id: String, site: String, title: String, pageviews: Double, rank: Double)

object TranslationRecommendations {
  val FEATURES = "features"
  val LABEL = "label"
  val PREDICTION = "prediction"
  val EXISTS = 1.0
  val NOT_EXISTS = 0.0

  case class Params(rawSitelinks: Option[File] = None,
                    rawPagecounts: Option[File] = None,
                    rawData: Option[File] = None,
                    parsedData: Option[File] = None,
                    featureData: Option[File] = None,
                    modelsDir: Option[File] = None,
                    outputDir: Option[File] = None,
                    targetWikis: Seq[String] = Seq(),
                    /* Actions */
                    parseRawData: Boolean = false,
                    extractFeatures: Boolean = false,
                    buildModels: Boolean = false,
                    scoreItems: Boolean = false)

  val argsParser = new OptionParser[Params]("Translation Recommendations") {
    head("Translation Recommendations", "")
    note("This job ranks items missing in languages by how much they would be read")
    help("help") text "Prints this usage text"

    /*
      mysql --host analytics-store.eqiad.wmnet wikidatawiki -e "select concat('Q', ips_item_id) as id, ips_site_id as site, replace(ips_site_page, ' ', '_') as title from wb_items_per_site join page on page_title = concat('Q', ips_item_id) where page_namespace = 0 and ips_site_id like '%wiki';" > sitelinks.tsv
      select
        concat('Q', ips_item_id) as id,
        ips_site_id as site,
        replace(ips_site_page, ' ', '_') as title
      from
        wb_items_per_site
      join
        page on page_title = concat('Q', ips_item_id)
      where
        page_namespace = 0
        and
        ips_site_id like '%wiki';
     */
    opt[File]("raw-sitelinks")
      .text("Raw sitelink data extracted from mysql")
      .optional()
      .valueName("<file>")
      .action((x, p) =>
        p.copy(rawSitelinks = Some(x))
      )
      .validate(f => if (f.exists()) success else failure("File does not exist"))

    /*
     * https://dumps.wikimedia.org/other/pagecounts-ez/merged/pagecounts-<year>-<month>-views-ge-5-totals.bz2
     */
    opt[File]("raw-pagecounts")
      .text("Raw pagecount data from wikimedia dumps")
      .optional()
      .valueName("<file>")
      .action((x, p) =>
        p.copy(rawPagecounts = Some(x))
      )
      .validate(f => if (f.exists()) success else failure("File does not exist"))

    opt[File]('r', "raw-data")
      .text("Raw data tsv of (id, site, title, pageviews) with header")
      .optional()
      .valueName("<file>")
      .action((x, p) =>
        p.copy(rawData = Some(x))
      )
      .validate(f => if (f.exists()) success else failure("File does not exist"))

    opt[File]('p', "parsed-data")
      .text("Parsed data of (id, site, title, pageviews)")
      .optional()
      .valueName("<path>")
      .action((x, p) =>
        p.copy(parsedData = Some(x))
      )
      .validate(f => if (f.exists()) success else failure("Path does not exist"))

    opt[File]('f', "feature-data")
      .text("Feature data of (id, (pageviews, rank, exists) * sites)")
      .optional()
      .valueName("<path>")
      .action((x, p) =>
        p.copy(featureData = Some(x))
      )
      .validate(f => if (f.exists()) success else failure("Path does not exist"))

    opt[File]('m', "models-dir")
      .text("Directory containing models named by target wiki")
      .optional()
      .valueName("<dir>")
      .action((x, p) => p.copy(modelsDir = Some(x)))
      .validate(f => if (f.exists() && f.isDirectory) success else failure("Dir is not valid"))

    opt[File]('o', "output-dir")
      .text("Directory to save the output of the steps")
      .optional()
      .valueName("<dir>")
      .action((x, p) =>
        p.copy(outputDir = Some(x))
      )
      .validate(f => if (f.exists() && f.isDirectory) success else failure("Dir is not valid"))

    opt[Unit]('a', "parse-raw-data")
      .text("Action to parse raw data")
      .optional()
      .action((_, p) => p.copy(parseRawData = true))

    opt[Unit]('x', "extract-features")
      .text("Action to extract features from parsed data")
      .optional()
      .action((_, p) => p.copy(extractFeatures = true))

    opt[Unit]('b', "build-models")
      .text("Action to build models")
      .optional()
      .action((_, p) => p.copy(buildModels = true))

    opt[Unit]('s', "score-items")
      .text("Action to score items")
      .optional()
      .action((_, p) => p.copy(scoreItems = true))

    opt[Seq[String]]('t', "target-wikis")
      .text("Target wikis to build models for")
      .optional()
      .valueName("<wiki1>,<wiki2>...")
      .action((x, p) => p.copy(targetWikis = x))

    checkConfig(c =>
      if (c.parseRawData && (c.rawData.isEmpty && (c.rawSitelinks.isEmpty || c.rawPagecounts.isEmpty))) {
        failure("Raw data not specified")
      } else if (!c.parseRawData && c.parsedData.isEmpty) {
        failure("Parsed data not specified")
      } else if (c.buildModels && !c.extractFeatures && c.featureData.isEmpty) {
        failure("Feature data not specified")
      } else {
        success
      }
    )
  }

  def main(args: Array[String]): Unit = {
    argsParser.parse(args, Params()) match {
      case Some(params) => {
        val spark = SparkSession
          .builder()
          .appName("TranslationRecommendations")
          .master("local[1]")
          .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        val timestamp = ZonedDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd-HHmmss"))

        val parsedData: Dataset[SitelinkPageviewsEntry] =
          if (params.parseRawData) {
            val p = parseRawData(spark, params.rawSitelinks, params.rawPagecounts, params.rawData)
            params.outputDir.foreach(o =>
              p.write.mode(SaveMode.ErrorIfExists).save(new File(o, timestamp + "_parsedData").getAbsolutePath))
            p
          } else {
            import spark.implicits._
            spark.read.load(params.parsedData.get.getAbsolutePath).as[SitelinkPageviewsEntry]
          }

        val featureData: DataFrame =
          if (params.extractFeatures) {
            val f = transformEntriesToFeatures(spark, parsedData.rdd)
            params.outputDir.foreach(o =>
              f.write.mode(SaveMode.ErrorIfExists).save(new File(o, timestamp + "_featureData").getAbsolutePath))
            f
          } else {
            spark.read.load(params.featureData.get.getAbsolutePath)
          }

        val regressor = new RandomForestRegressor()
          .setLabelCol(LABEL)
          .setFeaturesCol(FEATURES)
        val evaluator = new RegressionEvaluator()
          .setLabelCol(LABEL)
          .setPredictionCol(PREDICTION)
          .setMetricName("rmse")

        val sites = if (params.targetWikis.nonEmpty)
          params.targetWikis.toArray
        else
          parsedData.rdd.map(_.site).distinct().collect().sorted

        if (params.buildModels) {
          val modelsOutputDir = params.outputDir.map(o => new File(o, timestamp + "_models"))
          modelsOutputDir.foreach(o => o.mkdir())
          sites.foreach(target => {
            try {
              println(target)
              val workData: DataFrame = getWorkData(spark, featureData, target)
              val Array(trainingData, testData) = workData.randomSplit(Array(0.7, 0.3))
              val model = regressor.fit(trainingData)

              modelsOutputDir.foreach(o => model.write.save(new File(o, target).getAbsolutePath))

              val predictions = model.transform(testData)
              predictions.show(5)
              val rmse = evaluator.evaluate(predictions)
              println("Root Mean Squared Error (RMSE) on test data = " + rmse)
            } catch {
              case unknown: Throwable => println("Build model for " + target + " failed: " + unknown)
            }
          })
        }

        if (params.scoreItems) {
          val predictionsOutputDir = params.outputDir.map(o => new File(o, timestamp + "_predictions"))
          predictionsOutputDir.foreach(o => o.mkdir())
          sites.foreach(target => {
            try {
              println(target)
              val workData: DataFrame = getWorkData(spark, featureData, target, exists = false)
              val model = RandomForestRegressionModel.load(
                new File(params.modelsDir.get, target).getAbsolutePath)

              val predictions = model.transform(workData).select("id", PREDICTION)

              predictions.show(5)

              predictionsOutputDir.foreach(o =>
                predictions.write.mode(SaveMode.ErrorIfExists).csv(new File(o, target).getAbsolutePath))
            } catch {
              case unknown: Throwable => println("Score for " + target + " failed: " + unknown)
            }
          })
        }

        spark.stop()
      }
      case None => sys.exit(1)
    }
  }

  def parseRawData(
    spark: SparkSession,
    rawSitelinks: Option[File],
    rawPagecounts: Option[File],
    rawData: Option[File]
  ): Dataset[SitelinkPageviewsEntry] = {
    import spark.implicits._

    if (rawData.isEmpty) {
      val sitelinks = spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "DROPMALFORMED")
        .option("sep", "\t")
        .csv(rawSitelinks.get.getAbsolutePath)
        .as[SitelinkEntry]

      val pagecountSchema = StructType(Array(
        StructField("site", StringType),
        StructField("title", StringType),
        StructField("pageviews", DoubleType)
      ))

      val pagecounts = spark.read
        .format("csv")
        .option("quote", "\0")
        .option("escape", "\0")
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

      sitelinks.join(pagecounts, usingColumns = Seq("site", "title"))
        .as[SitelinkPageviewsEntry]
    } else {
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

  def getWorkData(spark: SparkSession, data: DataFrame, target: String, exists: Boolean = true): DataFrame = {
    val workData: DataFrame = data.filter(row =>
      row(row.fieldIndex("exists_" + target)) == (if (exists) EXISTS else NOT_EXISTS))

    import spark.implicits._
    val labeledData = workData.map(row =>
      (
        row.getString(row.fieldIndex("id")),
        row.getDouble(row.fieldIndex("rank_" + target)),
        /* Include everything except the target language's features */
        new DenseVector((
          (1 until row.fieldIndex("pageviews_" + target)).map(row.getDouble) ++
            (row.fieldIndex("exists_" + target) + 1 until row.length).map(row.getDouble)
          ).toArray)
      )
    ).rdd

    spark.createDataFrame(labeledData).toDF("id", LABEL, FEATURES)
  }

  case class PartitionKey(site: String, pageviews: Double) extends Ordered[PartitionKey] {
    override def compare(that: PartitionKey): Int =
      (this.site, this.pageviews) compare(that.site, that.pageviews)
  }

  class SitePartitioner[K <: PartitionKey](val numPartitions: Int)
    extends Partitioner {
    require(numPartitions >= 0,
      s"Number of partitions ($numPartitions) cannot be negative.")

    override def getPartition(key: Any): Int = {
      math.abs(key.asInstanceOf[K].site.hashCode()) % numPartitions
    }
  }

  def transformEntriesToFeatures(spark: SparkSession, entries: RDD[SitelinkPageviewsEntry]): DataFrame = {
    val sitesEntryCount = entries.map(_.site).countByValue()
    val sites = sitesEntryCount.toVector.map(_._1).sorted

    val partitioner = new SitePartitioner(16)
    val sortedGroupedEntries: RDD[(PartitionKey, (String, String))] = entries
      .map(e => (PartitionKey(e.site, e.pageviews), (e.id, e.title)))
      .repartitionAndSortWithinPartitions(partitioner)

    val rankedEntries = sortedGroupedEntries
      .mapPartitions((part: Iterator[(PartitionKey, (String, String))]) => {
        var currentSite: Option[String] = None
        var currentSiteEntryCount: Option[Double] = None
        var rank: Long = 0L
        part.map { case (key, (id, title)) => {
          if (currentSite.getOrElse(Nil) != key.site) {
            /* Initialize for site */
            rank = 0L
            currentSite = Some(key.site)
            currentSiteEntryCount = Some(sitesEntryCount(key.site).toDouble)
          }
          rank = rank + 1
          RankedEntry(id, key.site, title, key.pageviews, rank.toDouble / currentSiteEntryCount.get)
        }
        }
      })

    /*
     * Now that all entries have been ranked, combine the results into vectors grouped by id
    */
    val groupedEntries: RDD[(String, Iterable[RankedEntry])] = rankedEntries.groupBy(_.id)
    val itemMaps: RDD[(String, Map[String, (Double, Double)])] = groupedEntries.map { case (id, itemEntries) =>
      id -> itemEntries.map(entry => entry.site -> (entry.pageviews, entry.rank)).toMap
    }

    val structure = StructType(
      Array(StructField("id", StringType, nullable = false)) ++
        sites.flatMap(site => Seq(
          StructField("pageviews_" + site, DoubleType, nullable = false),
          StructField("rank_" + site, DoubleType, nullable = false),
          StructField("exists_" + site, DoubleType, nullable = false)
        )))

    val rows = itemMaps.map { case (id, itemMap) =>
      Row.fromSeq(Array(id) ++
        sites.flatMap(site => Seq(
          itemMap.getOrElse(site, (0.0, 0.0))._1, /* pageviews */
          itemMap.getOrElse(site, (0.0, 0.0))._2, /* rank */
          if (itemMap.contains(site)) EXISTS else NOT_EXISTS /* exists */
        ))
      )
    }

    spark.createDataFrame(rows, structure)
  }
}
