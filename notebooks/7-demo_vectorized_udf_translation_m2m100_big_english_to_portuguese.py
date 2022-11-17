# Databricks notebook source
pip install sentencepiece 

# COMMAND ----------

import mlflow
import pandas as pd
import time
from pyspark.sql import functions as F
from pyspark.sql.types import *

# identify the model we'll pull from the model registry
model_name = "m2m100_translation_transformer" 

# grab the uploaded sample data 
deltaDF = spark.read.table('kenjohnson_demo.default.eng_por')
deltaDF = deltaDF.toPandas()
count = deltaDF['English'].count()
df_source = pd.DataFrame({'id':[_ for _ in range(count)]})
df_source['content'] = deltaDF['English'].astype(str)

# COMMAND ----------

#Uncomment this to use only 64 values for debugging purposes.
#df_source = df_source[df_source['id'].isin(df_source['id'].value_counts().head(64).index)]

# COMMAND ----------

# convert pandas dataframe to Spark dataframe, and force Spark to partition the dataframe across all available executors
df_source_spark = spark.createDataFrame(df_source).repartition(spark.sparkContext.defaultParallelism).cache()

# COMMAND ----------

# inferencing function we'll distribute as a Pandas UDF
def translation_predictions_function(df):
    translation_loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    ##mlflow.pyfunc.PythonModel enforces a one argument predict function so we use a tuple to send in our src and target languages
    param_dict = {'src_lang': 'en', 'target_lang': 'pt', 'batch_size': 32}
    model_input = ([df, param_dict])
    return translation_loaded.predict(model_input)
    
# the Spark Pandas function API requires a return value schema
schema = StructType(
    [
      StructField("id", LongType(), True),
      StructField("content", StringType(), True),
      StructField("translation", StringType(), True)
    ]
)

inferencingStartTime = time.time()
# actual translation inference on the Spark dataframe
df_source_translation = (
    df_source_spark\
    .groupBy(F.spark_partition_id().alias("_pid"))\
    .applyInPandas(translation_predictions_function, schema)
).cache()
df_source_translation.write.mode("overwrite").format("noop").save()

# viewing the results dataframe in a Databricks notebook
display(df_source_translation)

# COMMAND ----------

inferencingEndTime = time.time()
totalInferencingTime = inferencingEndTime - inferencingStartTime
dataRowCount = df_source["id"].count()

print (f"{dataRowCount} Source phrases were translated. Inferencing phase took {totalInferencingTime} seconds on {df_source_spark.rdd.getNumPartitions()} total nodes.")

# COMMAND ----------

print(df_source_spark.rdd.getNumPartitions())
