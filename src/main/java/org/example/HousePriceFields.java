package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {

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

        csvData.describe().show();

        csvData = csvData.drop("id", "date",
                "grade", "waterfront", "view",
                "condition", "yr_renovated",
                "zipcode", "lat", "long", "sqft_lot", "yr_built", "sqft_lot15", "sqft_living15");
        csvData.describe().show();

        for(String col: csvData.columns()){
            System.out.println("Correlation between price and " + col + " :" + csvData.stat().corr("price", col));
        }

        for(String col1 : csvData.columns()){
            for(String col2 : csvData.columns())
                System.out.println("Correlation between " + col1 + " and " + col2 + " :" + csvData.stat().corr(col1, col2));
        }
    }
}
