from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when # Preprocess the text
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import NaiveBayes
# from sparknlp.annotator import LemmatizerModel
# import sparknlp
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import time

start = time.time()
print("hello")
# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Load the dataset
data = spark.read.csv("../spark-apps/training.1600000.processed.noemoticon.csv", header=None, inferSchema=True)

# Select and rename the relevant columns
data = data.selectExpr("_c0 as label", "_c5 as text")

# Remove unnecessary columns
data = data.select("text", "label")

# Convert text to lowercase
data = data.withColumn("text", lower(col("text")))

# Remove special characters and URLs
data = data.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))
data = data.withColumn("text", regexp_replace(col("text"), "http\\S+\\s?", ""))

data = data.withColumn("label", when(data.label == 4, 1).otherwise(data.label))

# Advanced Text Preprocessing
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
data = tokenizer.transform(data)

# Remove stopwords
stopwords = StopWordsRemover.loadDefaultStopWords("english")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stopwords)
data = remover.transform(data)

#Feature extraction
vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="features")
vectorizer_model = vectorizer.fit(data)
data = vectorizer_model.transform(data)

# Split the data into training and testing sets (80% training, 20% testing)
train, test = data.randomSplit([0.8, 0.2], seed=123)

nb = NaiveBayes(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[nb])
model = pipeline.fit(train)

# Make predictions on the test set
predictions = model.transform(test)

# Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)

# Save vectorizer_model and model
vectorizer_model.write().overwrite().save("../spark-apps/vectorizer_model")
model.write().overwrite().save("../spark-apps/model")

end = time.time()
print(end - start)
