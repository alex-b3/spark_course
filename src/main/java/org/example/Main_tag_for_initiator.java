package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.example.model.Build;
import org.example.service.BuildService;
import org.example.service.ScenarioService;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main_tag_for_initiator {
    public static void main(String[] args) throws Exception {

        BuildService buildService = new BuildService();
        ScenarioService scenarioService = new ScenarioService();

        List<Build> builds = buildService.readUsersFromCsv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/builds.csv");
//        List<Scenario> scenarios = scenarioService.readUsersFromCsv("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/scenarios.csv");


        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
        conf.set("spark.driver.bindAddress", "127.0.0.1");
        JavaSparkContext sc = new JavaSparkContext(conf);


        List<String> excludedInitiators = Arrays.asList("CD_AUTOMATION",
                "timer", "Daily_Automations",
                "Cloud_daily_Automations", "null",
                "place_holder", "NULL",
                "--upstream_jenkins", "automation.gituser",
                "TEST_CD_DYNAMIC_SLAVE", "Cron", "Timer");

        // Assuming builds and scenarios are collections of Build and Scenario objects
        JavaRDD<Build> buildsRDD = sc.parallelize(builds).filter(build -> !excludedInitiators.contains(build.getInitiator()));
//        JavaRDD<Scenario> scenariosRDD = sc.parallelize(scenarios);

        // Joining Build and Scenario Data
//        JavaPairRDD<Long, Tuple2<Scenario, Build>> joinedData = scenariosRDD
//                .mapToPair(scenario -> new Tuple2<>(scenario.getBuildId(), scenario))
//                .join(buildsRDD.mapToPair(build -> new Tuple2<>(build.getBuildId(), build)));



// =======================  Udemy tutorial example ====================
//        JavaRDD<String> initial = sc.textFile("/Users/alex.bichovsky/Desktop/spark_tutorial/src/main/resources/tutorial_data_1.csv")
//                .flatMap( word -> Arrays.asList(word.split("\t")).iterator())
//                .map( line -> line.replaceAll("[^a-zA-Z\\s]", "").trim().toLowerCase())
//                .filter( sentence -> sentence.length() > 0)
//                .filter( word -> !word.equals("null"))
//                .filter( word -> !word.contains("https"));
//
//        JavaPairRDD<String, Long> pairRdd = initial.mapToPair( word -> new Tuple2<>(word, 1L));
//        pairRdd.reduceByKey(Long::sum)
//                .mapToPair( tuple -> new Tuple2<>(tuple._2, tuple._1))
//                .sortByKey(false)
//                .collect().forEach(System.out::println);

//=========================  Showing tags ran by initiator example  =========================

//        List<String> excludedInitiators = Arrays.asList("CD_AUTOMATION",
//                "timer", "Daily_Automations",
//                "Cloud_daily_Automations", "null",
//                "place_holder", "NULL",
//                "--upstream_jenkins", "automation.gituser",
//                "TEST_CD_DYNAMIC_SLAVE", "Cron", "Timer");

        List<String> excludeTags = Arrays.asList("~@wip", "~@broken", "scenario_path");

        JavaRDD<Build> initial = sc.parallelize(builds)
                        .filter(build -> !excludedInitiators.contains(build.getInitiator()));

        JavaPairRDD<String, String> initiatorTagsPair = initial.flatMapToPair(build -> {
            String name = build.getInitiator();
            String[] tags = build.getTags().split(",");
            List<Tuple2<String, String>> pairs = new ArrayList<>();
            for (String tag : tags) {
                if (!excludeTags.contains(tag)) {
                    pairs.add(new Tuple2<>(name, tag.trim()));
                }
            }
            return pairs.iterator();
        });

        System.out.println("Example of the initiatorTagsPair:" + initiatorTagsPair.first());

        JavaPairRDD<Tuple2<String, String>, Integer> tagCounts = initiatorTagsPair.mapToPair(pair -> new Tuple2<>(pair, 1))
                .reduceByKey(Integer::sum);

        System.out.println("Example of the tagCounts:" + tagCounts.first());

        JavaPairRDD<String, Iterable<Tuple2<String, Integer>>> groupedAndSorted = tagCounts
                .mapToPair(pair -> new Tuple2<>(pair._1._1, new Tuple2<>(pair._1._2, pair._2)))
                .groupByKey()
                .mapValues(values -> {
                    List<Tuple2<String, Integer>> valueList = new ArrayList<>();
                    values.forEach(valueList::add);
                    valueList.sort((v1, v2) -> v2._2().compareTo(v1._2()));
                    return valueList;
                });

        groupedAndSorted.collect().forEach(result -> System.out.println(result._1() + " -> " + result._2()));
// ==================End of my data Example========================
        sc.close();
    }
}