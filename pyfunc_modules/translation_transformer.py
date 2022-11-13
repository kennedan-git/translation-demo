"""
Template for taking a Huggingface Transformer Named Entity Recognition pipeline and registering it as an MLflow model.
"""

import mlflow
import torch
import pandas as pd
import sentencepiece
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, pipeline

class TransformerTranslationModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
    
    def predict(self, df, srcLang, targetLang):
        """
        Inference logic that uses a Huggingface Transformer translation pipeline and generates a translation.
        Parameters
        ----------
        df: pandas.DataFrame
            data frame containing at least two columns: 
                'id'(Integer) - unique identifier for each row in the dataframe
                'content'(String) - text or document objects
            any other columns will be dropped in the returned results
        Returns
        -------
        pandas.DataFrame
            data frame containing three columns:
                'id'(Integer) - unique identifier for each row in the dataframe
                'content' (String) - original text or document objects
                'translation (String) - translated text or document objects
        """
        model_inputs = self._tokenizer(texts, return_tensors="pt")
        texts = df.content.values.tolist()
        ids = df.id.values.tolist()
        self._tokenizer.src_lang = srcLang #ex: 'en'
        
        gen_tokens = self._model.generate(**model_inputs, forced_bos_token_id=self._tokenizer.get_lang_id(targetLang))
        text_translations = self._tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        df_with_translations = pd.DataFrame({"id": ids, "content": texts, "translation": text_translations})
        return df_with_translations

def _load_pyfunc(data_path):
    """
    Required PyFunc custom loader module, following the second option in https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#workflows
    Note that although the Huggingface tokenizer and model instantiation calls can infer a web URL from a model card path, e.g. "dslim/bert-base-NER",
    MLflow Pyfunc loader modules treat the data_path argument as being in the local file system, i.e. in S3, ADLS or DBFS
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = M2M100Tokenizer.from_pretrained(data_path, padding=True)
    model = M2M100ForConditionalGeneration.from_pretrained(data_path)
    translation = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return TransformerTranslationModel(translation)