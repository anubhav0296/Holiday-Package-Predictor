from dataclasses import dataclass
import os
import sys
from components.model_trainer import ModelTrainer
import pandas as pd
import numpy as np

from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "Travel.csv")
    train_data_path:str = os.path.join("artifacts", "train_data.csv")
    test_data_path:str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()

    def initialize_data_ingestion(self):
        try:
            logging.info("Data Ingestion is initialized")
            # Read the raw data
            df = pd.read_csv("notebook/data/Travel.csv")

            # Create the folder where the raw data will be stored
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_data_path), exist_ok=True)

            # Export the data to csv
            df.to_csv(self.dataingestionconfig.raw_data_path, index=False, header=True)
            
            X = df.drop(['ProdTaken', 'NumberOfChildrenVisiting'], axis=1)
            y = df['ProdTaken']

            logging.info("Train test split has started")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            train_data = pd.concat([X_train, y_train], axis=1)
            train_data.to_csv(self.dataingestionconfig.train_data_path, index=False, header=True)
            logging.info("Train data exported successfully")

            test_data = pd.concat([X_test, y_test], axis=1)
            test_data.to_csv(self.dataingestionconfig.test_data_path, index=False, header=True)
            
            logging.info("Test data exported successfully")
            logging.info("Data is successfully exported")

            return (
                self.dataingestionconfig.train_data_path,
                self.dataingestionconfig.test_data_path
            )

        except Exception as e:
            logging("Exception has occured while data ingestion")
            raise CustomException(e, sys)    
    
if __name__ == "__main__":
    data_ing_obj = DataIngestion()
    train_data, test_data = data_ing_obj.initialize_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr, test_arr, _ = data_transformation_obj.initialize_data_transformation()

    modeltrainer = ModelTrainer()
    print(modeltrainer.get_best_model(train_arr,test_arr))