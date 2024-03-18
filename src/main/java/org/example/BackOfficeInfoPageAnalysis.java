package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import scala.collection.JavaConverters;
import scala.collection.mutable.WrappedArray;

import java.util.List;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;

public class BackOfficeInfoPageAnalysis {
    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("BackOfficeInfoPage Analysis")
                .master("local[*]")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();

        // Load the CSV files into DataFrames
        Dataset<Row> scenariosDf = spark.read().option("header", "true").csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/scenarios_for_steps.csv");
        Dataset<Row> stepsDf = spark.read().option("header", "true").csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/steps.csv");

        // Define the date range
        String startDate = "2023-11-06";
        String endDate = "2023-11-07";

        // Filter scenarios within the date range and count occurrences
        Dataset<Row> filteredScenariosCount = scenariosDf
                .filter(col("insert_time").geq(lit(startDate)).and(col("insert_time").leq(lit(endDate))))
                .groupBy("scenario_name", "example_number")
                .agg(count("*").alias("scenario_count"));

        // Filter steps for "BackOfficeInfoPage" within the date range
        Dataset<Row> backOfficeStepsDf = stepsDf.filter(col("step_name").like("%BackOfficeInfoPage%"))
                .filter(col("insert_time").geq(lit(startDate)).and(col("insert_time").leq(lit(endDate))));

        // Get the first occurrence of "BackOfficeInfoPage" step for each test_id
        Dataset<Row> firstBackOfficeStepDf = backOfficeStepsDf.groupBy("test_id")
                .agg(min("step_id").alias("first_step_id"));

        // Join to get all steps and then filter to keep those at or after the first "BackOfficeInfoPage" step
        Dataset<Row> followingStepsDf = stepsDf.join(firstBackOfficeStepDf, stepsDf.col("test_id").equalTo(firstBackOfficeStepDf.col("test_id")))
                .filter(stepsDf.col("step_id").geq(firstBackOfficeStepDf.col("first_step_id")))
                .join(scenariosDf, "test_id");

        // Group by scenario and example number, and collect step names
        Dataset<Row> stepsPerScenarioDf = followingStepsDf.groupBy("scenario_name", "example_number")
                .agg(collect_list("step_name").alias("steps"));

        // Define a UDF to concatenate list into a string
        UserDefinedFunction concatAsStr = udf((WrappedArray<String> steps) -> {
            List<String> javaSteps = JavaConverters.seqAsJavaListConverter(steps).asJava();
            return javaSteps.stream()
                    .collect(Collectors.joining(","));
        }, DataTypes.StringType);

        // Create a unique string identifier for each sequence of steps
        Dataset<Row> uniqueSequencesDf = stepsPerScenarioDf.withColumn("step_sequence", concatAsStr.apply(col("steps")))
                .select("scenario_name", "example_number", "step_sequence");

        // Join the unique sequences with the scenario counts
        Dataset<Row> finalDf = uniqueSequencesDf.join(filteredScenariosCount,
                        uniqueSequencesDf.col("scenario_name").equalTo(filteredScenariosCount.col("scenario_name"))
                                .and(uniqueSequencesDf.col("example_number").equalTo(filteredScenariosCount.col("example_number"))),
                        "left")
                .select(uniqueSequencesDf.col("scenario_name"), uniqueSequencesDf.col("example_number"),
                        filteredScenariosCount.col("scenario_count"), uniqueSequencesDf.col("step_sequence"))
                .distinct();

        // Display results
        for (Row row : finalDf.collectAsList()) {
            System.out.println("Scenario: " + row.getAs("scenario_name") + ", Example Number: " + row.getAs("example_number"));
            System.out.println("Scenario Count: " + row.getAs("scenario_count"));
            System.out.println("Step Sequence:");
            String stepSequence = row.getAs("step_sequence");
            for (String step : stepSequence.split(",")) {
                System.out.println(" - " + step);
            }
            System.out.println("-----------------------------------------------------");
        }


        spark.stop();
    }
}