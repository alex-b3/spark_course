package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class GymCompetitorsClustering {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("Gym Competitors")
                .config("spark.sql.warehouse.dir", "file:////Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/GymCompetition.csv");

        StringIndexer genderIndexer = new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");
        csvData = genderIndexer.fit(csvData).transform(csvData);

        OneHotEncoder genderEncoder = new OneHotEncoder();
        genderEncoder.setInputCols(new String[] {"GenderIndex"});
        genderEncoder.setOutputCols(new String[] {"GenderVector"});
        csvData = genderEncoder.fit(csvData).transform(csvData);

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {"GenderVector", "Age", "Height", "Weight", "NoOfReps"});
        Dataset<Row> inputData = vectorAssembler.setOutputCol("features").transform(csvData).select("features");

        KMeans kMeans = new KMeans();

        kMeans.setK(5);
        KMeansModel model = kMeans.fit(inputData);
        Dataset<Row> predictions = model.transform(inputData);
        Vector[] clusterCenters = model.clusterCenters();
        for(Vector v : clusterCenters){
            System.out.println(v);
        }

        predictions.groupBy("prediction").count().show();
        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        System.out.println("SSE is " + model.summary().trainingCost());
        System.out.println("Evaluator " + evaluator.evaluate(predictions));

        predictions.show();
    }

}
