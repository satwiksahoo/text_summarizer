from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.textSummarizer.logging import logger
import os
from datasets import load_from_disk 
import pandas as pd
import re
from datasets import Dataset
from src.textSummarizer.entity import dataTransformationConfig


class DataTransformation:
    def __init__(self , config : dataTransformationConfig):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name) 
    
    def preprocess_text(self , text):
        tokens = []
        text = re.sub(r'\r\n|\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        
        training_data = dataset['train']
        test_data = dataset['test']
        validation_data = dataset['validation']
        
        df_train = pd.DataFrame(training_data)
        df_test = pd.DataFrame(test_data)
        df_validation = pd.DataFrame(validation_data)
        
        df_train_small = df_train.sample(n= 6000 , random_state = 42).reset_index(drop = True)
        df_test_small = df_test.sample(n= 600 , random_state = 42).reset_index(drop = True)
        df_validation_small = df_validation.sample(n= 600 , random_state = 42).reset_index(drop = True)
         
        df_train_small['preprocess_text'] = df_train_small['dialogue'].apply(self.preprocess_text)
        df_test_small['preprocess_text'] = df_test_small['dialogue'].apply(self.preprocess_text)
        df_validation_small['preprocess_text'] = df_validation_small['dialogue'].apply(self.preprocess_text)
        
        train_dataset = Dataset.from_pandas(df_train_small)
        test_dataset = Dataset.from_pandas(df_test_small)
        validation_dataset = Dataset.from_pandas(df_validation_small)
        
        
        train_dataset.save_to_disk(os.path.join(self.config.root_dir , 'train_dataset'))
        test_dataset.save_to_disk(os.path.join(self.config.root_dir , 'test_dataset'))
        validation_dataset.save_to_disk(os.path.join(self.config.root_dir , 'validation_dataset'))
        
        
        
    
    
    
    