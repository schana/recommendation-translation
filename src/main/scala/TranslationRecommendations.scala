import scala.math.Ordered.orderingToOrdered
import org.apache.spark.Partitioner
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql._

case class SitelinkPageviewsEntry(id: String, site: String, title: String, pageviews: Double)

case class RankedEntry(id: String, site: String, title: String, pageviews: Double, rank: Double)

object TranslationRecommendations {
  val FEATURES = "features"
  val LABEL = "label"
  val PREDICTION = "prediction"
  val INPUT_FILE = "sitelinks-pagecounts.tsv"
  val BUILD_DATA = true
  val EXISTS = 1.0
  val NOT_EXISTS = 0.0

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = initializeSpark()

    val data: DataFrame = if (BUILD_DATA) getData(spark) else getData(spark, "./data")
    val workData: DataFrame = getWorkData(spark, data, "enwiki")

    val Array(trainingData, testData) = workData.randomSplit(Array(0.7, 0.3))

    val regressor = new RandomForestRegressor()
      .setLabelCol(LABEL)
      .setFeaturesCol(FEATURES)

    val model = regressor.fit(trainingData)
    val predictions = model.transform(testData)

    predictions.show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(LABEL)
      .setPredictionCol(PREDICTION)
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    if (BUILD_DATA) {
      data.write.mode(SaveMode.Overwrite).save("./data")
    }
    model.write.overwrite().save("./model")

    spark.stop()
  }

  def initializeSpark(): SparkSession = {
    val spark = SparkSession
      .builder()
      .appName("TranslationRecommendations")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark
  }

  def getData(spark: SparkSession, path: String): DataFrame = {
    spark.read.load(path)
  }

  def getData(spark: SparkSession): DataFrame = {
    val sitelinkPageviewsEntries: Dataset[SitelinkPageviewsEntry] = getSitelinkPageviewsEntries(spark)
    transformEntriesToFeatures(spark, sitelinkPageviewsEntries.rdd)
  }

  def getWorkData(spark: SparkSession, data: DataFrame, target: String): DataFrame = {
    val workData: DataFrame = data.filter(row => row(row.fieldIndex("exists_" + target)) == EXISTS)

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

  def getSitelinkPageviewsEntries(spark: SparkSession): Dataset[SitelinkPageviewsEntry] = {
    import spark.implicits._
    spark.read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .option("sep", "\t")
      .csv(INPUT_FILE)
      .as[SitelinkPageviewsEntry]
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
