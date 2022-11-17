# Databricks notebook source
pip install sentencepiece 

# COMMAND ----------

#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"

# COMMAND ----------

import mlflow
import pandas as pd
import time
from pyspark.sql import functions as F
from pyspark.sql.types import *

# identify the model we'll pull from the model registry
model_name = "m2m100_1B_translation_transformer" 

#start timing things 
start = time.time()

# pashto and english phrases the 'MEANING' column is in pashto, lets load that and translate it 
pashtoDF = spark.read.table('kenjohnson_demo.default.pashto_parallel_corpus')
pashtoDF = pashtoDF.toPandas()
count = pashtoDF['MEANING'].count()
display( count )
sourcelang = "ur" #urdu 
targetlang = "en" #english 
df_source = pd.DataFrame({'id':[_ for _ in range(count)]})
df_source['content'] = pashtoDF['MEANING'].astype(str)

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
    ##mlflow.pyfunc.PythonModel enforces a one argument predict function so we use a tuple to send in our params
    param_dict = {'src_lang': 'ur', 'target_lang': 'en', 'batch_size': 4}
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

# viewing the results dataframe in a Databricks notebook
display(df_source_translation)

# COMMAND ----------

count

# COMMAND ----------

# MAGIC %sql 
# MAGIC describe kenjohnson_demo.default.pashto_parallel_corpus

# COMMAND ----------

# MAGIC %sql 
# MAGIC ALTER TABLE kenjohnson_demo.default.pashto_parallel_corpus ADD COLUMNS (M2M1001BTranslation string)

# COMMAND ----------

# MAGIC %sql 
# MAGIC INSERT INTO kenjohnson_demo.default.pashto_parallel_corpus (M2M1001BTranslation) VALUES
# MAGIC     as select translation from kenjohnson_demo.default.pashto_parallel_corpustrans

# COMMAND ----------

Edf_source_translation.write.saveAsTable("kenjohnson_demo.default.pashto_parallel_corpustrans")

# COMMAND ----------

print(df_source_spark.rdd.getNumPartitions())
