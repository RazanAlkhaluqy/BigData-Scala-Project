
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.hadoop.fs.{FileSystem, Path}

object Preprocessing {
  def main(args: Array[String]): Unit = {
    // Step 1: Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("FoodDeliveryPreprocessing")
      .master("local[*]") // Run locally
      .getOrCreate()

    import spark.implicits._

    // Step 2: Load dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("Food_Delivery_Times.csv")

    df.show(5)

    // Step 3: Data Cleaning - Drop rows with missing values
    val cleanedDF = df.na.drop()

    // Check for null values after cleaning
    val nullCounts = cleanedDF.select(cleanedDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*)
    nullCounts.show()

    // Step 4: Data Transformation - Convert categorical columns to numerical using StringIndexer
    val categoricalCols = Array("Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type")
    val indexers = categoricalCols.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_Index")
    }

    val indexedDF = indexers.foldLeft(cleanedDF) { (tempDF, indexer) =>
      indexer.fit(tempDF).transform(tempDF)
    }

    // Step 5: Save the Preprocessed Data
    val outputDir = "C:/Users/razan/OneDrive/Desktop/BigData Project/preprocessed_data"

    indexedDF
      .coalesce(1)
      .write
      .option("header", "true")
      .mode("overwrite")
      .csv(outputDir)

    println(s"Preprocessed data saved to: $outputDir")

    // Stop SparkSession
    spark.stop()
  }
}
