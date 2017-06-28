package org.wikimedia.research.recommendation.job.translation

import java.io.File

import scopt.OptionParser

object ArgParser {

  case class Params(runLocally: Boolean = false,
                    rawSitelinks: Option[File] = None,
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

    opt[Unit]('l', "local")
      .text("Run Spark locally")
      .optional()
      .action((_, p) => p.copy(runLocally = true))

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

    opt[File]('r', "raw-data")
      .text("Raw data tsv of (id, site, title, pageviews) with header")
      .optional()
      .valueName("<file>")
      .action((x, p) =>
        p.copy(rawData = Some(x))
      )

    opt[File]('p', "parsed-data")
      .text("Parsed data of (id, site, title, pageviews)")
      .optional()
      .valueName("<path>")
      .action((x, p) =>
        p.copy(parsedData = Some(x))
      )

    opt[File]('f', "feature-data")
      .text("Feature data of (id, (pageviews, rank, exists) * sites)")
      .optional()
      .valueName("<path>")
      .action((x, p) =>
        p.copy(featureData = Some(x))
      )

    opt[File]('m', "models-dir")
      .text("Directory containing models named by target wiki")
      .optional()
      .valueName("<dir>")
      .action((x, p) => p.copy(modelsDir = Some(x)))

    opt[File]('o', "output-dir")
      .text("Directory to save the output of the steps")
      .optional()
      .valueName("<dir>")
      .action((x, p) =>
        p.copy(outputDir = Some(x))
      )

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
      } else if (c.scoreItems && c.modelsDir.isEmpty && (!c.buildModels || c.outputDir.isEmpty)) {
        failure("No models available for scoring. Either (build models and specify --output-dir) or (specify --models-dir)")
      } else {
        success
      }
    )
  }

  def parseArgs(args: Array[String]): Params = {
    argsParser.parse(args, Params()) match {
      case Some(params) => params
      case None => sys.exit(1)
    }
  }
}
