package org.wikimedia.research.recommendation.job.translation

import java.io.File

import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.wikimedia.research.recommendation.job.translation.ArgParser.Params

import scala.math.Ordered.orderingToOrdered

object FeatureExtractor {
  val log: Logger = LogManager.getLogger(FeatureExtractor.getClass)

  def extractFeatures(spark: SparkSession,
                      params: Params,
                      timestamp: String,
                      parsedData: Dataset[SitelinkPageviewsEntry]): DataFrame = {
    val featureData: DataFrame =
      if (params.extractFeatures) {
        log.info("Extracting feature data")
        val f = transformEntriesToFeatures(spark, parsedData.rdd)
        params.outputDir.foreach(o =>
          f.write.mode(SaveMode.ErrorIfExists).save(new File(o, timestamp + "_featureData").getAbsolutePath))
        f
      } else {
        log.info("Reading feature data")
        spark.read.load(params.featureData.get.getAbsolutePath)
      }
    featureData
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
          if (itemMap.contains(site)) Utils.EXISTS else Utils.NOT_EXISTS /* exists */
        ))
      )
    }

    spark.createDataFrame(rows, structure)
  }
}
