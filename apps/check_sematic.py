from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when # Preprocess the text
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover, CountVectorizerModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec
from pyspark.ml import PipelineModel

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Assuming you have new_data as a DataFrame with a 'text' column containing the new text data
new_data = spark.createDataFrame([["This is a positive sentence"],
                                  ["I'm feeling negative today"],
                                  ["Neutral statement here"]], ["text"])

# Advanced Text Preprocessing
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
new_data = tokenizer.transform(new_data)

# Remove stopwords
stopwords = StopWordsRemover.loadDefaultStopWords("english")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=stopwords)
new_data = remover.transform(new_data)

# Load vectorizer_model and model
loaded_vectorizer_model = CountVectorizerModel.load("../spark-apps/vectorizer_model")
loaded_model = PipelineModel.load("../spark-apps/model")

new_data = loaded_vectorizer_model.transform(new_data)
# Make predictions on the new data
predictions = loaded_model.transform(new_data)

predictions.show()

# Display the results
predictions.select("text","prediction").show(truncate=False)
