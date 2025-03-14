package edu.ucr.cs.cs167.skath004;

import java.util.HashMap;
import java.util.Map;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class Task_5 {
    public static void main(String[] args) {
        // Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("BirdCategoryPrediction")
                .master("local[*]") // Run locally
                .getOrCreate();

        // Load Parquet Dataset
        String filePath = "eBird_ZIP_1k.parquet";
        Dataset<Row> df = spark.read().parquet(filePath);

        // Register as a SQL table
        df.createOrReplaceTempView("birds");

        // Run Spark SQL Query to filter specific categories
        Dataset<Row> filteredDf = spark.sql(
                "SELECT COMMON_NAME, SCIENTIFIC_NAME, CATEGORY FROM birds " +
                        "WHERE CATEGORY IN ('species', 'form', 'issf', 'slash')"
        );

        filteredDf = filteredDf.dropDuplicates();

        Map<String, Double> fractions = new HashMap<>();
        fractions.put("species", 0.2);
        fractions.put("form", 1.0);
        fractions.put("issf", 1.0);
        fractions.put("slash", 1.0);

        Dataset<Row> balancedDf = filteredDf.stat().sampleBy("CATEGORY", fractions, 42);
        // Display filtered results (optional)
        System.out.println("Filtered Data Sample:");
        balancedDf.show(10, false);

        // Tokenizer for COMMON_NAME and SCIENTIFIC_NAME
        Tokenizer tokenizerCommon = new Tokenizer()
                .setInputCol("COMMON_NAME")
                .setOutputCol("COMMON_NAME_TOKENS");

        Tokenizer tokenizerScientific = new Tokenizer()
                .setInputCol("SCIENTIFIC_NAME")
                .setOutputCol("SCIENTIFIC_NAME_TOKENS");


        // HashingTF to convert tokens into numerical features
        HashingTF hashingTFCommon = new HashingTF()
                .setInputCol("COMMON_NAME_TOKENS")
                .setOutputCol("COMMON_NAME_FEATURES")
                .setNumFeatures(1000);

        HashingTF hashingTFScientific = new HashingTF()
                .setInputCol("SCIENTIFIC_NAME_TOKENS")
                .setOutputCol("SCIENTIFIC_NAME_FEATURES")
                .setNumFeatures(1000);

        // Vector Assembler to combine feature columns
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"COMMON_NAME_FEATURES", "SCIENTIFIC_NAME_FEATURES"})
                .setOutputCol("features");

        // Convert CATEGORY column into numerical labels
        StringIndexer indexer = new StringIndexer()
                .setInputCol("CATEGORY")
                .setOutputCol("label");

        // Logistic Regression Model
        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol("features")
                .setLabelCol("label");

        // Create Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new org.apache.spark.ml.PipelineStage[]{
                        tokenizerCommon, tokenizerScientific, hashingTFCommon, hashingTFScientific, assembler, indexer, lr
                });

        // Split data into training (80%) and testing (20%)
        Dataset<Row>[] splits = balancedDf.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train Model
        long startTime = System.currentTimeMillis();
        PipelineModel model = pipeline.fit(trainData);
        long trainingTime = System.currentTimeMillis() - startTime;

        // Make Predictions
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate Model Accuracy
        MulticlassClassificationEvaluator accuracyEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = accuracyEvaluator.evaluate(predictions);

        // Compute Precision and Recall
        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");

        double precision = precisionEvaluator.evaluate(predictions);
        double recall = recallEvaluator.evaluate(predictions);

        // Print results
        System.out.println("Model Training Time: " + trainingTime + " ms");
        System.out.println("Model Accuracy: " + accuracy);
        System.out.println("Model Precision: " + precision);
        System.out.println("Model Recall: " + recall);

        // Show sample predictions
        predictions.select("COMMON_NAME", "SCIENTIFIC_NAME", "CATEGORY", "label", "prediction").show(10, false);

        // Stop Spark Session
        spark.stop();
    }
}
