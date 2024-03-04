package org.example;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

/**
 * This class is used in the chapter late in the course where we analyse viewing figures.
 * You can ignore until then.
 */
public class ViewingFigures
{
	@SuppressWarnings("resource")
	public static void main(String[] args)
	{
		System.setProperty("hadoop.home.dir", "c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);

		SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
		conf.set("spark.driver.bindAddress", "127.0.0.1");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Use true to use hardcoded data identical to that in the PDF guide.
		boolean testMode = true;

		JavaPairRDD<Integer, Integer> viewData = setUpViewDataRdd(sc, testMode);
		JavaPairRDD<Integer, Integer> chapterData = setUpChapterDataRdd(sc, testMode);
		JavaPairRDD<Integer, String> titlesData = setUpTitlesDataRdd(sc, testMode);

		// Warmup
		JavaPairRDD<Integer, Integer> chapterCount = chapterData.mapToPair( row -> new Tuple2<>(row._2, 1)).reduceByKey(Integer::sum);
		chapterCount.collect().forEach(System.out::println);

		//Step 1
		viewData = viewData.distinct();

		//Step 2
		viewData = viewData.mapToPair( row -> new Tuple2<>(row._2, row._1));
		JavaPairRDD<Integer, Tuple2<Integer, Integer>> joinedRDD = viewData.join(chapterData);

		// Step 3
		JavaPairRDD<Tuple2<Integer, Integer>, Long> step3 = joinedRDD.mapToPair( row -> {
			Integer userId = row._2._1;
			Integer courseID = row._2._2;
			return new Tuple2<>(new Tuple2<>(userId, courseID), 1L);
		}).reduceByKey(Long::sum);

		//Step5

		JavaPairRDD<Integer, Long> step5 = step3.mapToPair(row -> new Tuple2<>(row._1._2, row._2));

		//Step 6
		JavaPairRDD<Integer, Tuple2<Long, Integer>> step6 = step5.join(chapterCount);


		//Step7

		JavaPairRDD<Integer, Double> step7 = step6.mapValues( value -> (double)value._1 / value._2);

		//Step8

		JavaPairRDD<Integer, Long> step8 = step7.mapValues(value -> {
			if(value > 0.9 ) return 10L;
			if( value > 0.5) return 4L;
			if( value > 0.25 ) return 2L;
			return 0L;
		});

		//Step9

		JavaPairRDD<Integer, Long> step9 = step8.reduceByKey(Long::sum);

		step9.collect().forEach(System.out::println);


		sc.close();
	}

	private static JavaPairRDD<Integer, String> setUpTitlesDataRdd(JavaSparkContext sc, boolean testMode) {

		if (testMode)
		{
			// (chapterId, title)
			List<Tuple2<Integer, String>> rawTitles = new ArrayList<>();
			rawTitles.add(new Tuple2<>(1, "How to find a better job"));
			rawTitles.add(new Tuple2<>(2, "Work faster harder smarter until you drop"));
			rawTitles.add(new Tuple2<>(3, "Content Creation is a Mug's Game"));
			return sc.parallelizePairs(rawTitles);
		}
		return sc.textFile("src/main/resources/viewing figures/titles.csv")
				                                    .mapToPair(commaSeparatedLine -> {
														String[] cols = commaSeparatedLine.split(",");
														return new Tuple2<>(new Integer(cols[0]), cols[1]);
				                                    });
	}

	private static JavaPairRDD<Integer, Integer> setUpChapterDataRdd(JavaSparkContext sc, boolean testMode) {

		if (testMode)
		{
			// (chapterId, (courseId, courseTitle))
			List<Tuple2<Integer, Integer>> rawChapterData = new ArrayList<>();
			rawChapterData.add(new Tuple2<>(96,  1));
			rawChapterData.add(new Tuple2<>(97,  1));
			rawChapterData.add(new Tuple2<>(98,  1));
			rawChapterData.add(new Tuple2<>(99,  2));
			rawChapterData.add(new Tuple2<>(100, 3));
			rawChapterData.add(new Tuple2<>(101, 3));
			rawChapterData.add(new Tuple2<>(102, 3));
			rawChapterData.add(new Tuple2<>(103, 3));
			rawChapterData.add(new Tuple2<>(104, 3));
			rawChapterData.add(new Tuple2<>(105, 3));
			rawChapterData.add(new Tuple2<>(106, 3));
			rawChapterData.add(new Tuple2<>(107, 3));
			rawChapterData.add(new Tuple2<>(108, 3));
			rawChapterData.add(new Tuple2<>(109, 3));
			return sc.parallelizePairs(rawChapterData);
		}

		return sc.textFile("src/main/resources/viewing figures/chapters.csv")
													  .mapToPair(commaSeparatedLine -> {
															String[] cols = commaSeparatedLine.split(",");
															return new Tuple2<>(new Integer(cols[0]), new Integer(cols[1]));
													  	});
	}

	private static JavaPairRDD<Integer, Integer> setUpViewDataRdd(JavaSparkContext sc, boolean testMode) {

		if (testMode)
		{
			// Chapter views - (userId, chapterId)
			List<Tuple2<Integer, Integer>> rawViewData = new ArrayList<>();
			rawViewData.add(new Tuple2<>(14, 96));
			rawViewData.add(new Tuple2<>(14, 97));
			rawViewData.add(new Tuple2<>(13, 96));
			rawViewData.add(new Tuple2<>(13, 96));
			rawViewData.add(new Tuple2<>(13, 96));
			rawViewData.add(new Tuple2<>(14, 99));
			rawViewData.add(new Tuple2<>(13, 100));
			return  sc.parallelizePairs(rawViewData);
		}

		return sc.textFile("src/main/resources/viewing figures/views-*.csv")
				     .mapToPair(commaSeparatedLine -> {
				    	 String[] columns = commaSeparatedLine.split(",");
				    	 return new Tuple2<>(new Integer(columns[0]), new Integer(columns[1]));
				     });
	}
}
