
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import java.io.File
import java.nio.file.{Files, Paths, StandardCopyOption}
import scala.collection.JavaConverters._

object MergedPreprocessing {
  def main(args: Array[String]): Unit = {
    // Step 1: Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("FoodDeliveryPreprocessing")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Step 2: Load dataset
    val inputFilePath = "Food_Delivery_Times.csv" // Ensure correct path
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputFilePath)

    df.show(5)

    // Step 3: Data Cleaning - Drop rows with missing values
    val cleanedDF = df.na.drop()

    // Check for null values after cleaning
    val nullCounts = cleanedDF.select(cleanedDF.columns.map(c => sum(col(c).isNull.cast("int")).alias(c)): _*)
    nullCounts.show()

    // Step 4: Add Speed Column (Speed = Distance / Time_of_Delivery)
    val processedDF = cleanedDF.withColumn("Speed", col("Distance_km") / col("Delivery_Time_min"))

    // Step 5: Data Transformation - Convert categorical columns to numerical using StringIndexer
    val categoricalCols = Array("Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type")
    val indexers = categoricalCols.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_Index")
    }

    val indexedDF = indexers.foldLeft(processedDF) { (tempDF, indexer) =>
      indexer.fit(tempDF).transform(tempDF)
    }

    // Step 6: Drop original categorical columns after transformation
    val finalDF = indexedDF.drop(categoricalCols: _*)

    // Step 7: Additional Processing - Dropping Unnecessary Columns and Adding Speed in km/min
    val dfWithSpeed = finalDF.withColumn("Speed_km_per_min", col("Distance_km") / col("Delivery_Time_min"))

   
   // Save the file
val tempOutputDir = "C:\\temp\\BigDataOutput"
dfWithSpeed.coalesce(1)
  .write.option("header", "true")
  .mode("overwrite")
  .csv(tempOutputDir)

// Rename the single part file
val dir = new File(tempOutputDir)
val partFile = dir.listFiles().filter(_.getName.startsWith("part-")).head
val newFile = new File(tempOutputDir + "\\FinalData.csv")
Files.move(partFile.toPath, newFile.toPath, StandardCopyOption.REPLACE_EXISTING)

println(s"File saved as: $newFile")
   
    /* Step 8: Save the final processed data
    val outputDir = "C:\\Users\\razan\\OneDrive - King Suad University\\BigData Project\\BigData-Scala-Project"
    dfWithSpeed
     .coalesce(1)
  .write
  .option("header", "true")
  .mode("overwrite")
  .csv(outputDir)

println(s"Final cleaned and processed data saved to: $outputDir")
*/

    // Stop SparkSession
    spark.stop()
  }
}
