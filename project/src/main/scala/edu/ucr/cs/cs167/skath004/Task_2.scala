package edu.ucr.cs.cs167.skath004

import edu.ucr.cs.bdlab.beast.geolite.{Feature, IFeature}
import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


import scala.collection.Map


object Task_2 {
  def main(args: Array[String]): Unit = {
    // Initialize Spark context
    val conf = new SparkConf().setAppName("Beast Example")
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")


    val spark: SparkSession.Builder = SparkSession.builder().config(conf)


    val sparkSession: SparkSession = spark.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    SparkSQLRegistration.registerUDT
    SparkSQLRegistration.registerUDF(sparkSession)


    val inputFile: String = args(0)
    import edu.ucr.cs.bdlab.beast._


    val keyword: String = args(1)


    sparkSession.read
      .parquet(inputFile).createOrReplaceTempView("observations")
    sparkSession.sql(
      s"""
     SELECT ZipCode, SUM(OBSERVATION_COUNT) As observation
     FROM observations
     GROUP BY ZIPCode
   """).createOrReplaceTempView("total")


    //    df.printSchema()
    sparkSession.sql(
      s"""
     SELECT ZipCode, SUM(OBSERVATION_COUNT) As species_observation
     FROM observations
     WHERE CATEGORY = 'species'
     GROUP BY ZIPCode
   """).createOrReplaceTempView("species")


    sparkSession.sql(
      s"""
     SELECT t.ZIPCode, t.observation, s.species_observation,
     (s.species_observation * 1.0/t.observation) AS ratio
     FROM total t
     LEFT JOIN species s ON t.ZIPCode = s.ZIPCode
   """).createOrReplaceTempView("total_join")


    val zipcodesDF = sparkSession.read.format("shapefile")
      .load("tl_2018_us_zcta510.zip")  // Update with the correct path
    zipcodesDF.createOrReplaceTempView("zipcodes_with_geometry")


    sparkSession.sql(
      s"""
   SELECT z.ZCTA5CE10 AS Zip, z.geometry, r.ratio
   FROM zipcodes_with_geometry z
   JOIN total_join r ON r.ZipCode = z.ZCTA5CE10
 """).createOrReplaceTempView("final_data")


    val finalData: DataFrame = sparkSession.sql("SELECT * FROM final_data")
    val spatialRDD = finalData.toSpatialRDD
    spatialRDD.coalesce(1)  // Ensure single output file
      .saveAsShapefile("eBirdZIPCodeRatio")
    val t2 = System.nanoTime()
  }
}
