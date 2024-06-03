import org.apache.log4j.{Level, Logger, PropertyConfigurator}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StandardScaler, StopWordsRemover, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
// import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

object MovieRecommendation {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("MovieRecommendation")
      .master("local[*]")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Configure logging
    PropertyConfigurator.configure("/Users/andresrocha/Downloads/CSC369/Project/src/main/resources/log4j.properties")

    import spark.implicits._

    // Load dataset
    val df = spark.read.option("header", "true")
      .csv("top_1000_popular_movies_tmdb.csv")
      .withColumn("vote_average", $"vote_average".cast("double"))
      .withColumn("vote_count", $"vote_count".cast("double"))
      .withColumn("popularity", $"popularity".cast("double"))
      .withColumn("runtime", $"runtime".cast("double"))

    // Replace null or empty strings in 'overview' with a default value
    val defaultOverview = "No overview available"
    val cleanedRDD = df.rdd.map { row =>
      val overview = row.getAs[String]("overview")
      val cleanedOverview = if (overview == null || overview.trim.isEmpty) defaultOverview else overview
      Row.fromSeq(row.toSeq.updated(row.fieldIndex("overview"), cleanedOverview))
    }

    val schema = df.schema
    val cleanedDF = spark.createDataFrame(cleanedRDD, schema)

    // Calculate medians or means for numerical columns
    val voteAverageMedian = cleanedDF.stat.approxQuantile("vote_average", Array(0.5), 0.001).head
    val voteCountMedian = cleanedDF.stat.approxQuantile("vote_count", Array(0.5), 0.001).head
    val popularityMedian = cleanedDF.stat.approxQuantile("popularity", Array(0.5), 0.001).head
    val runtimeMedian = cleanedDF.stat.approxQuantile("runtime", Array(0.5), 0.001).head

    // Fill missing values
    val filledDF = cleanedDF.na.fill(Map(
      "vote_average" -> voteAverageMedian,
      "vote_count" -> voteCountMedian,
      "popularity" -> popularityMedian,
      "runtime" -> runtimeMedian
    ))

    // Text preprocessing
    val tokenizer = new RegexTokenizer()
      .setInputCol("overview")
      .setOutputCol("tokens")
      .setPattern("\\W")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_tokens")

    val hashingTF = new HashingTF()
      .setInputCol("filtered_tokens")
      .setOutputCol("raw_features")
      .setNumFeatures(10000) //The vector is capped at 10000 for easy comparison

    val idf = new IDF()
      .setInputCol("raw_features")
      .setOutputCol("tfidf_features")

    // Combine features
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf_features", "vote_average", "vote_count", "popularity", "runtime"))
      .setOutputCol("features")
      .setHandleInvalid("skip")  // Skips rows with any invalid data

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
      .setWithStd(true)
      .setWithMean(false)

    // Pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, idf, assembler, scaler))

    val model = pipeline.fit(filledDF)
    val processedDF = model.transform(filledDF)

    // Function to compute cosine similarity
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      val dotProduct = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }.sum
      val normA = math.sqrt(v1.toArray.map(x => x * x).sum)
      val normB = math.sqrt(v2.toArray.map(x => x * x).sum)
      dotProduct / (normA * normB)
    }

    // Function to find nearest neighbors
    def findNearestNeighbors(query: String, numericalData: Array[Double], k: Int = 5): DataFrame = {
      val queryDF = Seq((query, numericalData(0), numericalData(1), numericalData(2), numericalData(3))).toDF("overview", "vote_average", "vote_count", "popularity", "runtime")
      val queryProcessedDF = model.transform(queryDF)
      val queryFeatures = queryProcessedDF.select("scaled_features").first().getAs[Vector]("scaled_features")

      val similarities = processedDF.select("title", "overview", "vote_average", "vote_count", "popularity", "runtime", "scaled_features").as[(String, String, Double, Double, Double, Double, Vector)].map {
        case (title, overview, voteAvg, voteCount, popularity, runtime, features) =>
          val similarity = cosineSimilarity(queryFeatures, features)
          (title, overview, voteAvg, voteCount, popularity, runtime, similarity)
      }


      val nearestNeighbors = similarities.sort($"_7".desc).take(k)
      spark.createDataFrame(nearestNeighbors).toDF("title", "overview", "vote_average", "vote_count", "popularity", "runtime", "similarity")
    }

    // Example query
    val query = "After reuniting with Gwen Stacy, Brooklyn’s full-time, friendly neighborhood Spider-Man is catapulted across the Multiverse, where he encounters the Spider Society, a team of Spider-People charged with protecting the Multiverse’s very existence. But when the heroes clash on how to handle a new threat, Miles finds himself pitted against the other Spiders and must set out on his own to save those he loves most."
    val numericalData = Array(8.8, 1160, 2859.047, 140)
    val nearestNeighbors = findNearestNeighbors(query, numericalData)

    nearestNeighbors.show()
    spark.stop()
  }
}
