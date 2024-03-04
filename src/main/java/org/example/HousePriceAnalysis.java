package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class HousePriceAnalysis {

    public static void main(String[] args) {
        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("House Price Analysis")
                .config("spark.sql.warehouse.dir","file:////Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read().option("header", true)
                .option("inferSchema", true)
                .csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/kc_house_data.csv");

        csvData.printSchema();
        csvData.show();

        csvData = csvData.withColumn("sqft_above_percentage", col("sqft_above")
                .divide(col("sqft_living"))).withColumnRenamed("price", "label");

        Dataset<Row>[] dataSplit = csvData.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> trainingAndTestData = dataSplit[0];
        Dataset<Row> holdOutData = dataSplit[1];

        StringIndexer conditionIndexer = new StringIndexer();
        conditionIndexer.setInputCol("condition");
        conditionIndexer.setOutputCol("conditionIndex");

        StringIndexer gradeIndexer = new StringIndexer();
        gradeIndexer.setInputCol("grade");
        gradeIndexer.setOutputCol("gradeIndex");

        StringIndexer zipCodeIndexer = new StringIndexer();
        zipCodeIndexer.setInputCol("zipcode");
        zipCodeIndexer.setOutputCol("zipcodeIndex");

        OneHotEncoder encoder = new OneHotEncoder();
        encoder.setInputCols(new String[] {"conditionIndex", "gradeIndex","zipcodeIndex"});
        encoder.setOutputCols(new String[] {"conditionVector","gradeVector","zipcodeVector"});

        VectorAssembler vectorAssembler = new VectorAssembler()
        .setInputCols(new String []{"bedrooms", "bathrooms",
                "sqft_living", "sqft_above_percentage", "floors","conditionVector", "gradeVector", "zipcodeVector", "waterfront"})
        .setOutputCol("features");

        LinearRegression linearRegression = new LinearRegression();
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();

        ParamMap[] paramMaps = paramGridBuilder.addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1})
                .build();

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(linearRegression)
                        .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                                .setEstimatorParamMaps(paramMaps)
                                        .setTrainRatio(0.8);

        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{ conditionIndexer, gradeIndexer, zipCodeIndexer, encoder, vectorAssembler, trainValidationSplit});
        PipelineModel pipelineModel =  pipeline.fit(trainingAndTestData);
        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");

        TrainValidationSplitModel model =  (TrainValidationSplitModel)pipelineModel.stages()[5];
        LinearRegressionModel linearRegressionModel = (LinearRegressionModel)model.bestModel();

        System.out.println("The training data r2 value is + " + linearRegressionModel.summary().r2() + " and the RMSE is " + linearRegressionModel.summary().rootMeanSquaredError());
        linearRegressionModel.transform(holdOutResults).show();

        System.out.println("The testing data r2 value is + " + linearRegressionModel.evaluate(holdOutResults).r2() + " and the RMSE is " + linearRegressionModel.evaluate(holdOutResults).rootMeanSquaredError());

        System.out.println("coefficients " + linearRegressionModel.coefficients() + " intercept " + linearRegressionModel.intercept());
        System.out.println("reg param: " + linearRegressionModel.getRegParam() + " elastic net param " + linearRegressionModel.getElasticNetParam());
     }
}
