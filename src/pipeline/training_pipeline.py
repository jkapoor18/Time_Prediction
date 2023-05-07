import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import warnings

#if __name__=='__main__':
def train_model():
    warnings.filterwarnings('ignore')
    obj=DataIngestion()
    raw_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(raw_data_path)
    model_trainer=ModelTrainer()
    best_model_output = model_trainer.initate_model_training(train_arr,test_arr)
    return best_model_output