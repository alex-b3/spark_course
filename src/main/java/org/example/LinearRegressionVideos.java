package org.example;


import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class LinearRegressionVideos {
    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("Case study Price Analysis")
                .config("spark.sql.warehouse.dir","file:////Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read().option("header", true)
                .option("inferSchema", true)
                .csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00001-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00002-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00003-d55d9fed-7427-4d23-aa42-495275510f78.csv");

        csvData = csvData.filter(col("is_cancelled").equalTo("true"))
                .drop(col("observation_date"), col("is_cancelled"))
                .na().fill(0);

        csvData = csvData.withColumnRenamed("next_month_views", "label");

        StringIndexer payMethodIndexer = new StringIndexer();
        payMethodIndexer.setInputCol("payment_method_type")
                .setOutputCol("payMethodIndexer");

        StringIndexer periodIndexer = new StringIndexer();
        periodIndexer.setInputCol("rebill_period_in_months")
                .setOutputCol("periodIndexer");

        StringIndexer countryIndexer = new StringIndexer();
        countryIndexer.setInputCol("country")
                .setOutputCol("countryIndexer");

        OneHotEncoder encoder = new OneHotEncoder();
        encoder.setInputCols(new String[] {"payMethodIndexer", "countryIndexer", "periodIndexer"})
                .setOutputCols(new String[] {"payMethodVector","countryVector", "periodVector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] {"firstSub", "age", "all_time_views", "last_month_views",
                "payMethodVector", "countryVector", "periodVector"})
                .setOutputCol("assembledFeatures");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("assembledFeatures")
                .setOutputCol("features")
                .setWithStd(true)
                .setWithMean(true);

        Dataset<Row>[] dataSplit = csvData.randomSplit(new double[] {0.9, 0.1});
        Dataset<Row> trainingAndTestData = dataSplit[0];
        Dataset<Row> holdOutData = dataSplit[1];

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMaps = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.05, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.25, 0.5, 0.75, 1})
                .addGrid(linearRegression.maxIter(), new int[]{50, 100, 200})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(linearRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMaps)
                .setNumFolds(5);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{ payMethodIndexer, periodIndexer, countryIndexer, encoder, vectorAssembler, scaler, crossValidator});

        PipelineModel pipelineModel =  pipeline.fit(trainingAndTestData);
        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults = holdOutResults.drop("prediction");

        CrossValidatorModel model =  (CrossValidatorModel)pipelineModel.stages()[6];
        LinearRegressionModel linearRegressionModel = (LinearRegressionModel)model.bestModel();

        System.out.println("The training data r2 value is + " + linearRegressionModel.summary().r2() + " and the RMSE is " + linearRegressionModel.summary().rootMeanSquaredError());
        linearRegressionModel.transform(holdOutResults).show(20);

        System.out.println("The testing data r2 value is + " + linearRegressionModel.evaluate(holdOutResults).r2() + " and the RMSE is " + linearRegressionModel.evaluate(holdOutResults).rootMeanSquaredError());

        System.out.println("coefficients " + linearRegressionModel.coefficients() + " intercept " + linearRegressionModel.intercept());
        System.out.println("reg param: " + linearRegressionModel.getRegParam() + " elastic net param " + linearRegressionModel.getElasticNetParam());

//        csvData.printSchema();
//        csvData.show();
    }
}
