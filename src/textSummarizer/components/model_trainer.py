from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.textSummarizer.logging import logger
import os
from datasets import load_from_disk 
import pandas as pd
import re
from datasets import Dataset
from src.textSummarizer.entity import ModelTrainerConfig
# Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
import torch




class ModelTrainer:
    def __init__(self , config : ModelTrainerConfig):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name) 
    
    
    def preprocess_fun(example):
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        input = tokenizer(example['preprocess_text'] , padding = 'max_length' ,truncation = True , max_length=512)
        target = tokenizer(example['summary'] , padding = 'max_length' ,truncation = True , max_length=150)
        input['labels'] = target['input_ids']

        return input
    
    def model_training(self):
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model =     AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        
        dataset = load_from_disk(self.config.data_path)
        
        training_args = TrainingArguments(
          output_dir='./results',
          num_train_epochs=self.params.num_train_epochs,
          per_device_train_batch_size=self.params.per_device_train_batch_size,     # smallest batch size
          per_device_eval_batch_size=self.params.per_device_eval_batch_size,
          weight_decay=self.params.weight_decay,
          logging_steps=self.params.logging_steps,
          eval_steps=self.params.eval_steps,
          save_steps=self.params.save_steps,
          gradient_accumulation_steps=self.params.gradient_accumulation_steps,     # no accumulation
          eval_strategy=self.params.eval_strategy,
          save_total_limit=self.params.save_total_limit,
          logging_dir='./logs',
    # Uncomment below to force CPU if MPS still crashes
    # no_cuda=True,
             )
        
        
        train_dataset = pd.DataFrame(load_from_disk(self.config.data_path_train))
        test_dataset = pd.DataFrame(load_from_disk(self.config.data_path_test))
        validation_dataset = pd.DataFrame(load_from_disk(self.config.data_path_validation))
        
        train_dataset_ = train_dataset.apply(preprocess_fun , axis = 1)
        test_dataset_ = test_dataset.apply(preprocess_fun ,axis = 1)
        validation_dataset_ = validation_dataset.apply(preprocess_fun , axis = 1)
        
        
        
        
        trainer = Trainer(
              model=model,
              args=training_args,
              train_dataset=train_dataset_,
              eval_dataset=validation_dataset_
             )
        
        trainer.train()
        
        model.save_pretrained(os.path.join(self.config.root_dir , 'bart_model'))
        
        tokenizer.save_pretrained(os.path.join(self.config.root_dir , 'tokenizer'))
        
        
        