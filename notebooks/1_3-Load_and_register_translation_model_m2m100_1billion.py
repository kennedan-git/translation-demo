# Databricks notebook source
pip install sentencepiece

# COMMAND ----------

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, pipeline
model_name = "facebook/m2m100_1.2B"
base_data_path = "/Users/Kenneth.Johnson/Documents/ml/huggingface/m2m100_1_2B/"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
nlp = pipeline("translation", model=model, tokenizer=tokenizer)

# COMMAND ----------

nlp.save_pretrained(base_data_path)

# COMMAND ----------

from logging import shutdown
import mlflow
import json
import pandas as pd
import shutil
import os

code_root_path = "/Workspace/Repos/kenneth.johnson@databricks.com/translation-demo/"

f = open(os.path.join(code_root_path ,"conda_envs/conda_translation.json"))
conda_env = json.load(f)
f.close()

def test_conda_env():
    assert isinstance(conda_env, dict)

with mlflow.start_run(run_name="translation_m2m100_1B") as run:
    mlflow.pyfunc.log_model(artifact_path="translation", 
        loader_module="translation_transformer", 
        conda_env=conda_env, code_path=[os.path.join(code_root_path,"pyfunc_modules/translation_transformer.py")], 
        data_path=base_data_path.replace("dbfs:", "/dbfs")
    )

translation_loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/translation")

def test_model_load():
    assert(isinstance(translation_loaded, mlflow.pyfunc.PyFuncModel))

test_df = pd.DataFrame({"id":[1,2,3], "content": ["Abraham Lincoln cut down a cherry tree", "Florida has nice beaches", "Elon Musk owns Tesla"]})

param_dict = {'src_lang': 'en', 'target_lang': 'pt', 'batch_size': 1}
model_input = ([test_df, param_dict])
results = translation_loaded.predict(model_input)

def test_model_inference():
    assert(results.columns.values.tolist() == ['id', 'content', 'translation'] and results.shape[0]==3)

pd.set_option('display.max_colwidth', None)
print(results.head())

# clean up
#mlflow.delete_run(run.info.run_id)
# used in local environment tests. Not necessary when running in Databricks
#shutil.rmtree(f"mlruns/0/{run.info.run_id}/", ignore_errors=True)

# COMMAND ----------

# clean up
#mlflow.delete_run(run.info.run_id)

# COMMAND ----------

result = test_model_load()
results 
