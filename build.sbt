name := "MovieRecommendation"

version := "0.1"

scalaVersion := "2.12.10"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.1.2",
  "org.apache.spark" %% "spark-sql" % "3.1.2",
  "org.apache.spark" %% "spark-mllib" % "3.1.2",
  "org.apache.spark" %% "spark-hive" % "3.1.2",
  "org.scalanlp" %% "breeze" % "1.2",
  "org.scalanlp" %% "breeze-natives" % "1.2",
  "org.apache.lucene" % "lucene-core" % "8.9.0",
  "org.apache.lucene" % "lucene-analyzers-common" % "8.9.0"
)
