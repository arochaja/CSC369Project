import org.apache.log4j.{Level, Logger, PropertyConfigurator}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler

object MovieRecommendation2 {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("MovieRecommendation").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Configure logging
    PropertyConfigurator.configure("/Users/andresrocha/Downloads/CSC369/Project/src/main/resources/log4j.properties")

    // Load dataset
    val data = sc.textFile("top_1000_popular_movies_tmdb.csv")
    val header = data.first()
    val rows = data.filter(row => row != header).map(_.split(",").map(_.trim))

    // Replace null or empty strings in 'overview' with a default value
    val defaultOverview = "No overview available"
    val cleanedRows = rows.map { row =>
      val overview = if (row(3) == null || row(3).isEmpty) defaultOverview else row(3)
      Array(row(0), row(1), row(2), overview, row(4), row(5), row(6), row(7))
    }

    // Helper function to parse double values with error handling
    def parseDoubleSafe(s: String, default: Double): Double = {
      try {
        s.toDouble
      } catch {
        case _: NumberFormatException => default
      }
    }

    // Calculate medians for numerical columns
    def median(array: RDD[Double]): Double = {
      val sortedArray = array.sortBy(x => x)
      val count = sortedArray.count()
      if (count % 2 == 0) {
        (sortedArray.take((count / 2).toInt).last + sortedArray.take((count / 2 + 1).toInt).last) / 2
      } else {
        sortedArray.take((count / 2 + 1).toInt).last
      }
    }

    val voteAverageMedian = median(cleanedRows.map(row => parseDoubleSafe(row(4), 0.0)))
    val voteCountMedian = median(cleanedRows.map(row => parseDoubleSafe(row(5), 0.0)))
    val popularityMedian = median(cleanedRows.map(row => parseDoubleSafe(row(6), 0.0)))
    val runtimeMedian = median(cleanedRows.map(row => parseDoubleSafe(row(7), 0.0)))

    // Fill missing values
    val filledRows = cleanedRows.map { row =>
      val voteAverage = if (row(4) == "null" || row(4).isEmpty) voteAverageMedian else parseDoubleSafe(row(4), voteAverageMedian)
      val voteCount = if (row(5) == "null" || row(5).isEmpty) voteCountMedian else parseDoubleSafe(row(5), voteCountMedian)
      val popularity = if (row(6) == "null" || row(6).isEmpty) popularityMedian else parseDoubleSafe(row(6), popularityMedian)
      val runtime = if (row(7) == "null" || row(7).isEmpty) runtimeMedian else parseDoubleSafe(row(7), runtimeMedian)
      (row(0), row(1), row(2), row(3), voteAverage, voteCount, popularity, runtime)
    }

    // Text preprocessing
    val hashingTF = new HashingTF(10000)
    val tf = hashingTF.transform(filledRows.map(_._4.split("\\W").toSeq))
    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf)

    // Combine features
    val features = filledRows.zip(tfidf).map {
      case ((title, _, _, _, voteAverage, voteCount, popularity, runtime), tfidfFeatures) =>
        val denseVector = Vectors.dense(Array(voteAverage, voteCount, popularity, runtime))
        val combinedFeatures = Vectors.dense(tfidfFeatures.toArray ++ denseVector.toArray)
        (title, combinedFeatures)
    }

    val scaler = new StandardScaler(withMean = false, withStd = true).fit(features.map(_._2))
    val scaledFeatures = features.map { case (title, vector) => (title, scaler.transform(vector)) }

    // Function to compute cosine similarity
    def cosineSimilarity(v1: Vector, v2: Vector): Double = {
      val dotProduct = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }.sum
      val normA = math.sqrt(v1.toArray.map(x => x * x).sum)
      val normB = math.sqrt(v2.toArray.map(x => x * x).sum)
      dotProduct / (normA * normB)
    }

    // Function to find nearest neighbors
    def findNearestNeighbors(query: String, numericalData: Array[Double], k: Int = 5): Array[(String, Double)] = {
      val queryVector = Vectors.dense(hashingTF.transform(query.split("\\W").toSeq).toArray ++ numericalData)
      val scaledQueryVector = scaler.transform(queryVector)

      val similarities = scaledFeatures.map { case (title, features) =>
        val similarity = cosineSimilarity(scaledQueryVector, features)
        (title, similarity)
      }

      similarities.top(k)(Ordering.by(_._2))
    }

    // Example query
    val query = "After reuniting with Gwen Stacy, Brooklyn’s full-time, friendly neighborhood Spider-Man is catapulted across the Multiverse, where he encounters the Spider Society, a team of Spider-People charged with protecting the Multiverse’s very existence. But when the heroes clash on how to handle a new threat, Miles finds himself pitted against the other Spiders and must set out on his own to save those he loves most."
    val numericalData = Array(8.8, 1160, 2859.047, 140)
    val nearestNeighbors = findNearestNeighbors(query, numericalData)

    nearestNeighbors.foreach { case (title, similarity) =>
      println(s"Title: $title, Similarity: $similarity")
    }

    sc.stop()
  }
}
