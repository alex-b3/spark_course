package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;

public class Roman {
    public static void main(String[] args) {

        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("BackOfficeInfoPage Analysis")
                .master("local[*]")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();

        Dataset<Row> scenariosDf = spark.read().option("header", "true").csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/scenarios_for_steps.csv");
        Dataset<Row> stepsDf = spark.read().option("header", "true").csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/steps.csv");

        String startDate = "2023-11-01";
        String endDate = "2023-11-07";

//        Dataset<Row> backOfficeStepsDf = stepsDf.filter(col("step_name").like("%front of the SM queue%"))
//                .filter(col("insert_time").geq(lit(startDate)).and(col("insert_time").leq(lit(endDate))));

        Dataset<Row> backOfficeStepsDf = stepsDf.filter(col("test_id").equalTo("43575204"));

        backOfficeStepsDf.show(false);

        System.out.println("Steps count: " + backOfficeStepsDf.count());

    }
}
