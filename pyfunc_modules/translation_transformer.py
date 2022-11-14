"""
Takes a Huggingface Transformer m2m100 Translation pipeline and registers it as an MLflow model.
"""

import mlflow
import torch
import pandas as pd
import sentencepiece
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, pipeline

class TransformerTranslationModel(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self._pipe = pipeline

    def translate(self, txt):
        if (len(txt) > 300): 
            return "Text too long."
        encoded_txt = self._pipe.tokenizer(txt, return_tensors="pt")
        encoded_txt = encoded_txt.to(self._pipe.device)
        generated_tokens = self._pipe.model.generate(**encoded_txt, forced_bos_token_id=self._pipe.tokenizer.get_lang_id("pt"))
        txt_translation = self._pipe.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return txt_translation

    #def generate(self, **encoded_txt): 
     #   #self._pipe.model.to(self._pipe.device)
      #  generated_txt = self._pipe.model.generate(**encoded_txt, forced_bos_token_id=self._pipe.tokenizer.get_lang_id("pt"))
       # return generated_txt
    
    def predict(self, df):
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

        texts = df.content.values.tolist()
        ids = df.id.values.tolist()
        #encoded_text = self._tokenizer(texts[0], return_tensors="pt")
        self._pipe.tokenizer.src_lang = "en" #ex: 'en'

        translation = df["content"]
        translation = translation.map(self.translate)
        translations = translation.tolist()

        df_with_translations = pd.DataFrame({"id": ids, "content": texts, "translation": translations})
        torch.cuda.empty_cache()
        return df_with_translations

def _load_pyfunc(data_path):
    """
    Required PyFunc custom loader module, following the second option in https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#workflows
    Note that although the Huggingface tokenizer and model instantiation calls can infer a web URL from a model card path, e.g. "dslim/bert-base-NER",
    MLflow Pyfunc loader modules treat the data_path argument as being in the local file system, i.e. in S3, ADLS or DBFS
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = M2M100Tokenizer.from_pretrained(data_path)
    model = M2M100ForConditionalGeneration.from_pretrained(data_path).to(device)
    translation = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return TransformerTranslationModel(translation)