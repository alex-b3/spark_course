package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class LogisticRegressionVideos {

    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("Case study Price Analysis")
                .config("spark.sql.warehouse.dir", "file:////Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read().option("header", true)
                .option("inferSchema", true)
                .csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00001-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00002-d55d9fed-7427-4d23-aa42-495275510f78.csv",
                        "/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/case study/part-r-00003-d55d9fed-7427-4d23-aa42-495275510f78.csv");

        // 1- customers watched NO videos, 0 - customers watched some videos
        csvData = csvData
                .withColumn("next_month_views", when (col("next_month_views").$greater(0), 0)
                .otherwise(1));

        csvData = csvData.filter(col("is_cancelled").equalTo("true"))
                .drop(col("observation_date"), col("is_cancelled"))
                .na().fill(0);

        csvData = csvData.withColumnRenamed("next_month_views", "label");

        csvData.printSchema();
        csvData.show();

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
                .setOutputCol("features");

        Dataset<Row>[] dataSplit = csvData.randomSplit(new double[] {0.9, 0.1});
        Dataset<Row> trainingAndTestData = dataSplit[0];
        Dataset<Row> holdOutData = dataSplit[1];

        LogisticRegression logisticRegression = new LogisticRegression();

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramMaps = paramGridBuilder
                .addGrid(logisticRegression.regParam(), new double[]{0.01, 0.05, 0.1, 0.5})
                .addGrid(logisticRegression.elasticNetParam(), new double[]{0, 0.25, 0.5, 0.75, 1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit();
        trainValidationSplit.setEstimator(logisticRegression)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMaps)
                .setTrainRatio(0.9);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{ payMethodIndexer, periodIndexer, countryIndexer, encoder, vectorAssembler, trainValidationSplit});

        PipelineModel pipelineModel =  pipeline.fit(trainingAndTestData);
        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults = holdOutResults.drop("prediction").drop("rawPrediction").drop("probability");

        TrainValidationSplitModel model =  (TrainValidationSplitModel)pipelineModel.stages()[5];
        LogisticRegressionModel logisticRegressionModel = (LogisticRegressionModel)model.bestModel();

        System.out.println("The accuracy score " + logisticRegressionModel.summary().accuracy());

        System.out.println("coefficients " + logisticRegressionModel.coefficients()
                + " intercept " + logisticRegressionModel.intercept());

        System.out.println("reg param " + logisticRegressionModel.getRegParam()
                + " elastic net param " + logisticRegressionModel.getElasticNetParam());

        LogisticRegressionSummary summary = logisticRegressionModel.evaluate(holdOutResults);
        logisticRegressionModel.transform(holdOutResults).show(1000);

        double truePositives = summary.truePositiveRateByLabel()[1];
        double falsePositives = summary.falsePositiveRateByLabel()[0];

        System.out.println("For the holdout data, the likelihood of a positive being correct is "
                + truePositives / (truePositives + falsePositives));
        System.out.println("The holdout data accuracy is " + summary.accuracy());

        logisticRegressionModel.transform(holdOutResults).groupBy("label", "prediction").count().show();


    }
}
