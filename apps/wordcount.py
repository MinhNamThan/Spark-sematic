from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read input text file
text_file = spark.read.text("../spark-apps/fileDemo.txt")
# text_file = spark.read.text("test.txt")
# Split each line into words
words = text_file.rdd.flatMap(lambda line: line.value.split(" "))

# Map each word to a tuple (word, 1)
word_tuples = words.map(lambda word: (word, 1))

# Reduce by key to count the occurrences of each word
word_counts = word_tuples.reduceByKey(lambda a, b: a + b)
# Print the word counts
for word, count in word_counts.collect():
    print(f"{word}: {count}")

# Stop the SparkSession
spark.stop()
