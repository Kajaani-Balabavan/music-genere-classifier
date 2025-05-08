from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, IDF,
    StringIndexer, IndexToString
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("MusicGenreClassification").getOrCreate()

# Ensure output directories exist
os.makedirs("../data", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# Step 2: Load datasets from root directory
mendeley_df = spark.read.csv("../mendeley_dataset.csv", header=True, inferSchema=True)
student_df = spark.read.csv("../Student_dataset.csv", header=True, inferSchema=True)

# Step 3: Select relevant columns
columns = ["artist_name", "track_name", "release_date", "genre", "lyrics"]
mendeley_df = mendeley_df.select(*columns)
student_df = student_df.select(*columns)

# Step 4: Merge datasets
merged_df = mendeley_df.union(student_df).dropna(subset=["genre", "lyrics"])

# Step 4.1: Save merged dataset
merged_output_path = "../data/merged_dataset.csv"
merged_df.coalesce(1).write.option("header", True).mode("overwrite").csv(merged_output_path)

# Step 5: Split data
train, test = merged_df.randomSplit([0.8, 0.2], seed=42)

# Step 6: Preprocessing stages
tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="genre", outputCol="label").fit(train)
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_genre", labels=label_indexer.labels)

# Step 7: Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0,
    family="multinomial"
)

# Step 8: Build full pipeline
pipeline = Pipeline(stages=[
    tokenizer,
    remover,
    vectorizer,
    idf,
    label_indexer,
    lr,
    label_converter
])

# Step 9: Train the model
model = pipeline.fit(train)

# Step 10: Save models
model.write().overwrite().save("../models/trained_model_pipeline")
label_indexer.write().overwrite().save("../models/label_indexer_model")

# Step 11: Evaluation
predictions = model.transform(test)

if predictions.count() == 0:
    raise ValueError("Predictions DataFrame is empty.")

# Accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Step 12: Export predictions to CSV
output_path = "../data/predicted_results.csv"
export_columns = ["artist_name", "track_name", "release_date", "lyrics", "genre", "predicted_genre"]
predictions.select(*export_columns).coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)
print(f"\nPredicted results saved to: {output_path}")

# Step 13: Confusion matrix
preds_and_labels = predictions.select("prediction", "label") \
                              .rdd.map(lambda row: (float(row["prediction"]), float(row["label"])))
metrics = MulticlassMetrics(preds_and_labels)
conf_matrix = metrics.confusionMatrix().toArray()

# Save confusion matrix to CSV
conf_df = pd.DataFrame(conf_matrix, columns=label_indexer.labels, index=label_indexer.labels)
conf_df.index.name = "Actual / Predicted"
conf_matrix_path = "../data/confusion_matrix.csv"
conf_df.to_csv(conf_matrix_path)
print(f"\nConfusion matrix saved to: {conf_matrix_path}")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

labels = label_indexer.labels
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha="right")
plt.yticks(tick_marks, labels)

thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, int(conf_matrix[i, j]),
                 ha="center", va="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("../data/confusion_matrix_plot.png")
plt.show()
