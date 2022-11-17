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

#start timing things 
start = time.time()

# pashto and english phrases the 'MEANING' column is in pashto, lets load that and translate it 
deltaDF = spark.read.table('kenjohnson_demo.default.train_ruen_df_short')
deltaDF = deltaDF.toPandas()
count = deltaDF['original'].count()
display( count )
sourcelang = "ru"  
targetlang = "en" 
df_source = pd.DataFrame({'id':[_ for _ in range(count)]})
df_source['content'] = deltaDF['original'].astype(str)

# COMMAND ----------

#Uncomment this to use only 64 values for debugging purposes.
#df_source = df_source[df_source['id'].isin(df_source['id'].value_counts().head(64).index)]

# COMMAND ----------

# convert pandas dataframe to Spark dataframe, and force Spark to partition the dataframe across all available executors
df_source_spark = spark.createDataFrame(df_source).repartition(spark.sparkContext.defaultParallelism).cache()


# COMMAND ----------



# COMMAND ----------

# inferencing function we'll distribute as a Pandas UDF
def translation_predictions_function(df):
    translation_loaded = mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")
    ##mlflow.pyfunc.PythonModel enforces a one argument predict function so we use a tuple to send in our params
    param_dict = {'src_lang': 'ru', 'target_lang': 'en', 'batch_size': 8}
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
#ArrayType(StringType())
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
