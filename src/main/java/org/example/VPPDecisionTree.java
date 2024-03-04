package org.example;

import org.apache.jute.Index;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import javax.xml.crypto.Data;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;

public class VPPDecisionTree {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("Case study Price Analysis")
                .config("spark.sql.warehouse.dir", "file:////Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .master("local[*]").getOrCreate();

        spark.udf().register("countryGrouping", countryGrouping, DataTypes.StringType);

        Dataset<Row> csvData = spark.read().option("header", true)
                .option("inferSchema", true)
                .csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/vppFreeTrials.csv");

        csvData = csvData.withColumn("country", callUDF("countryGrouping", col("country")))
                .withColumn("label", when(col("payments_made").geq(1), lit(1)).otherwise(lit(0)));

        StringIndexer countryIndexer = new StringIndexer();
        csvData = countryIndexer.setInputCol("country").setOutputCol("countryIndex")
                .fit(csvData).transform(csvData);

        new IndexToString()
                .setInputCol("countryIndex")
                .setOutputCol("value")
                .transform(csvData.select("countryIndex").distinct())
                .show();

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] {"countryIndex", "rebill_period", "chapter_access_count", "seconds_watched"})
                        .setOutputCol("features");

        Dataset<Row> inputData = vectorAssembler.transform(csvData).select("label", "features");

        Dataset<Row>[] dataSplit = inputData.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> trainingAndTestData = dataSplit[0];
        Dataset<Row> holdOutData = dataSplit[1];

        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();
        decisionTreeClassifier.setMaxDepth(3);
        DecisionTreeClassificationModel model = decisionTreeClassifier.fit(trainingAndTestData);

        Dataset<Row> predictions = model.transform(holdOutData);
        predictions.show();

        System.out.println(model.toDebugString());

        MulticlassClassificationEvaluator evaluator  = new MulticlassClassificationEvaluator();
        System.out.println("The accuracy of the model is: " + evaluator.setMetricName("accuracy").evaluate(predictions));

        RandomForestClassifier randomForest = new RandomForestClassifier();
        randomForest.setMaxDepth(3);
        RandomForestClassificationModel rfModel = randomForest.fit(trainingAndTestData);
        Dataset<Row> rfPredictions = rfModel.transform(holdOutData);
        rfPredictions.show();

        System.out.println(rfModel.toDebugString());
        System.out.println("The accuracy of the Random Forest model is: " + evaluator.setMetricName("accuracy").evaluate(rfPredictions));


//
//        inputData.show();
//
//        csvData.show();

    }

    public static UDF1<String,String> countryGrouping = new UDF1<String,String>() {

        @Override
        public String call(String country) throws Exception {
            List<String> topCountries =  Arrays.asList("GB","US","IN","UNKNOWN");
            List<String> europeanCountries =  Arrays.asList("BE","BG","CZ","DK","DE","EE","IE","EL","ES","FR","HR","IT","CY","LV","LT","LU","HU","MT","NL","AT","PL","PT","RO","SI","SK","FI","SE","CH","IS","NO","LI","EU");

            if (topCountries.contains(country)) return country;
            if (europeanCountries .contains(country)) return "EUROPE";
            else return "OTHER";
        }

    };
}
