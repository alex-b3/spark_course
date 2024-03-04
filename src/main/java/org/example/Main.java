package org.example;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.sql.*;
import org.example.model.Build;
import org.example.model.BuildFactory;

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.not;

public class Main {

    @SuppressWarnings("resourse")
    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("Tutorial Spark SQL")
                .master("local[*]")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();

        Dataset<Row> biglogSet = spark.read().option("header", true).csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/biglog.txt");

        biglogSet.createOrReplaceTempView("logging_table");

        // Load data from your source (could be a DB or CSV file or other sources)
        Dataset<Row> builds = spark.read().option("header", true).csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/builds.csv");
        Dataset<Row> scenarios = spark.read().option("header", true).csv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/scenarios.csv");

        // Explode the tags from Builds
        Dataset<Row> explodedBuilds = builds
                .withColumn("run_tag", explode(split(col("tags"), ",")))
                .select("initiator", "build_id", "run_tag");

// Explode the tags from Scenarios for accurate joining
        Dataset<Row> explodedScenarios = scenarios
                .withColumn("scenario_tag", explode(split(col("tags"), ",")))
                .select("build_id", "test_id", "result", "scenario_tag");

// Join explodedBuilds with explodedScenarios on build_id and tags
        Dataset<Row> joined = explodedBuilds
                .join(explodedScenarios,
                        explodedBuilds.col("build_id").equalTo(explodedScenarios.col("build_id"))
                                .and(explodedBuilds.col("run_tag").equalTo(trim(explodedScenarios.col("scenario_tag")))), "inner")
                .drop(explodedScenarios.col("build_id"))
                .filter(not(col("run_tag").isin("~@wip", "~@broken", "scenario_path")))
                .filter(not(col("initiator").isin("CD_AUTOMATION", "timer", "Daily_Automations", "Cloud_daily_Automations",
                        "null", "place_holder", "NULL", "--upstream_jenkins", "automation.gituser",
                        "TEST_CD_DYNAMIC_SLAVE", "Cron", "Timer")));

        // Aggregate to get the metrics
        Dataset<Row> aggregated = joined.groupBy("initiator", "run_tag")
                .agg(
                        count("*").alias("totalScenarios"),
                        sum(when(col("result").equalTo("1"), 1).otherwise(0)).alias("passedScenarios"),
                        sum(when(col("result").equalTo("0"), 1).otherwise(0)).alias("failedScenarios"),
                        collect_set("build_id").alias("build_ids")
                );

        // Calculate the pass rate
        aggregated = aggregated.withColumn("passRate", col("passedScenarios").divide(col("totalScenarios"))).orderBy("run_tag");

//        // Convert to JSON and print
//        ObjectMapper mapper = new ObjectMapper();
//        Dataset<String> jsonDataset = aggregated.toJSON();
//
//        jsonDataset.write().text("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/result");

        aggregated.explain();

//        jsonDataset.foreachPartition(partition -> {
//            partition.forEachRemaining(jsonStr -> {
//                try {
//                    Object json = mapper.readValue(jsonStr, Object.class);
//                    System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(json));
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            });
//        });

        spark.stop();
    }
}

//============================================Starting of Spark SQL example==========================================================
// Create temporary views
//        builds.createOrReplaceTempView("Builds");
//        scenarios.createOrReplaceTempView("Scenarios");
//
//        String sql = "WITH exploded AS (" +
//                "SELECT b.initiator, " +
//                "b.build_id, " +
//                "EXPLODE(SPLIT(b.tags, ',')) as run_tag " +
//                "FROM Builds b " +
//                ") " +
//                "SELECT initiator, run_tag as tag, " +
//                "COLLECT_SET(e.build_id) as build_ids, " +
//                "COUNT(DISTINCT s.test_id) as totalScenarios, " +
//                "SUM(CASE WHEN s.result = 1 THEN 1 ELSE 0 END) as passedScenarios, " +
//                "SUM(CASE WHEN s.result = 0 THEN 1 ELSE 0 END) as failedScenarios " +
//                "FROM exploded e " +
//                "JOIN Scenarios s ON e.build_id = s.build_id " +
//                "WHERE e.initiator NOT IN ('CD_AUTOMATION','timer', 'Daily_Automations', 'Cloud_daily_Automations', 'null', 'place_holder', 'NULL','--upstream_jenkins', 'automation.gituser','TEST_CD_DYNAMIC_SLAVE', 'Cron', 'Timer') " +
//                "AND CONCAT(',', s.tags, ',') LIKE CONCAT('%,', e.run_tag, ',%') " + // Making sure it matches the tag exactly without partial matches
//                "AND NOT (e.run_tag LIKE '%~@wip%' OR e.run_tag LIKE '%~@broken%' OR e.run_tag LIKE '%scenario_path%') " +
//                "GROUP BY initiator, run_tag";
//
//        Dataset<Row> result = spark.sql(sql);
//
//        // Compute average pass rate
//        result = result.withColumn("passRate", functions.col("passedScenarios").divide(functions.col("totalScenarios")));
//
//        ObjectMapper mapper = new ObjectMapper();
//        Dataset<String> jsonDataset = result.toJSON();
//
//        jsonDataset.foreachPartition(partition -> {
//            partition.forEachRemaining(jsonStr -> {
//                try {
//                    Object json = mapper.readValue(jsonStr, Object.class);
//                    System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(json));
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            });
//        });
//======================================================================================================
//        result.show();

//        Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/builds.csv");
//        List<String> excludeTags = Arrays.asList("~@wip", "~@broken", "scenario_path");
//
//
//        List<String> excludedInitiators = Arrays.asList("CD_AUTOMATION",
//                "timer", "Daily_Automations",
//                "Cloud_daily_Automations", "null",
//                "place_holder", "NULL",
//                "--upstream_jenkins", "automation.gituser",
//                "TEST_CD_DYNAMIC_SLAVE", "Cron", "Timer");
//
//        Dataset<Row> filteredDataset = dataset.filter(
//                not(col("initiator").isin((Object[]) excludedInitiators.toArray(new String[0]))));
//
//        filteredDataset.show();
//
//        Dataset<Build> builds = filteredDataset.map((MapFunction<Row, Build>) BuildFactory::createBuildFromRow, Encoders.bean(Build.class));
//        System.out.println("Number of Builds is: " + builds.count());
