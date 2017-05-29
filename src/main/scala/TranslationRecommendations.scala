import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

case class SitelinkPageviewsEntry(id: String, site: String, title: String, pageviews: Double)

case class RankedEntry(id: String, site: String, title: String, pageviews: Double, rank: Double)

object TranslationRecommendations {
  val FEATURES = "features"
  val LABEL = "label"
  val PREDICTION = "prediction"
  val INPUT_FILE = "lite-sitelinks-pagecounts.tsv"

  def main(args: Array[String]): Unit = {
    val spark = initializeSpark()

    val data = getData(spark, "enwiki")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

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

    spark.stop()
  }

  def initializeSpark(): SparkSession = {
    val spark = SparkSession
      .builder()
      .appName("TranslationRecommendations")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setCheckpointDir(".")
    spark.sparkContext.setLogLevel("WARN")
    spark
  }

  def getData(spark: SparkSession, target: String): DataFrame = {
    val sitelinkPageviewsEntries = getSitelinkPageviewsEntries(spark)
    val data = transformEntriesToFeatures(spark, sitelinkPageviewsEntries.rdd)
    val workData = data.filter(row => row(row.fieldIndex("exists_" + target)) == 1.0)

    import spark.implicits._
    val labeledData = workData.map(row =>
      (
        row.getString(row.fieldIndex("id")),
        row.getDouble(row.fieldIndex("rank_" + target)),
        new DenseVector(
          ((1 until row.fieldIndex("rank_" + target)).map(row.getDouble) ++
            (row.fieldIndex("rank_" + target) + 1 until row.length).map(row.getDouble)).toArray
        )
      )
    ).rdd
    labeledData.checkpoint()

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

  def transformEntriesToFeatures(spark: SparkSession, entries: RDD[SitelinkPageviewsEntry]): DataFrame = {
    val sites = entries.map(_.site).distinct.collect.sorted

    val groupedSites = entries.groupBy(_.site).collect()
    val rankedEntries = groupedSites.map { case (site, items) => rank(spark, items) }
      .fold(spark.sparkContext.emptyRDD)((a, b) => a ++ b)

    val groupedEntries = rankedEntries.groupBy(_.id)
    val itemMaps = groupedEntries.map { case (id, itemEntries) =>
      id -> itemEntries.map(entry => entry.site -> (entry.pageviews, entry.rank)).toMap
    }

    val structure =
      StructType(
        Array(StructField("id", StringType, nullable = false)) ++
          sites.map(site => StructField("pageviews_" + site, DoubleType, nullable = false)) ++
          sites.map(site => StructField("rank_" + site, DoubleType, nullable = false)) ++
          sites.map(site => StructField("exists_" + site, DoubleType, nullable = false))
      )
    val rows = itemMaps.map { case (id, itemMap) =>
      Row.fromSeq(Array(id) ++
        sites.map(itemMap.getOrElse(_, (0.0, 0.0))._1) ++ // pageviews
        sites.map(itemMap.getOrElse(_, (0.0, 0.0))._2) ++ // rank
        sites.map(site => if (itemMap.contains(site)) 1.0 else 0.0) // exists
      )
    }

    spark.createDataFrame(rows, structure)
  }

  def rank(spark: SparkSession, items: Iterable[SitelinkPageviewsEntry]): RDD[RankedEntry] = {
    val entries = spark.sparkContext.parallelize(items.toSeq)
    val ranked = entries.sortBy(_.pageviews).zipWithIndex()
    val count = ranked.count() - 1
    ranked.map { case (item, rank) =>
      RankedEntry(item.id, item.site, item.title, item.pageviews, rank.toDouble / count.toDouble)
    }
  }
}
