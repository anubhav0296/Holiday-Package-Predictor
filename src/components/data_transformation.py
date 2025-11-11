import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    processed_file_path:str = os.path.join("artifacts", "transformed_train_data.pkl")
    # transformed_test_data_path:str = os.path.join("artifacts", "transformed_test_data.pxl")

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def get_data_transformation_object(self, num_cols, cat_cols):
        try:
            # Then Create num_pipeline and cat_pipeline with steps
            num_pipeline = Pipeline(
                steps= [
                    ("Imputation", SimpleImputer(strategy="median")),
                    ("Standardization", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputation", SimpleImputer(strategy="most_frequent")),
                    ("Encoding", OneHotEncoder()),
                    ("Standardization", StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical and categorical pipeline has been defined")

            # Then use both num_cols and num_pipeline in ColumnTransformerß
            pre_processor = ColumnTransformer(
                [("num_transformation", num_pipeline, num_cols),
                ("cat_transformation", cat_pipeline, cat_cols)]
            )

            logging.info("Pre-processing object is created successfully")

            return pre_processor

        except Exception as e:
            logging.info("Exception has occured while getting transformed object")
            raise CustomException(e, sys)

    def initialize_data_transformation(self):
        try:
            logging.info("Data Transformation has started")
            train_data_df = pd.read_csv("artifacts/train_data.csv")
            test_data_df = pd.read_csv("artifacts/test_data.csv")

            target_col = ['ProdTaken']

            logging.info(f"{train_data_df[:5]}")            

            X_train = train_data_df.drop('ProdTaken', axis=1)
            y_train = train_data_df['ProdTaken']

            X_test = test_data_df.drop('ProdTaken', axis=1)
            y_test = test_data_df['ProdTaken']

            num_cols = [col for col in X_train.columns if X_train[col].dtype != "object"]
            cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

            processor_obj = self.get_data_transformation_object(num_cols, cat_cols)

            # X_train = train_data_df.drop('ProdTaken', axis=1)

            X_train_transformed = processor_obj.fit_transform(X_train)
            X_test_transformed = processor_obj.transform(X_test)

            logging.info("X_train and X_test data is transformed now")

            # Concatenating the train array and output variables
            train_transformed_arr = np.c_[X_train_transformed, y_train]
            test_transformed_arr = np.c_[X_test_transformed, y_test]

            save_object(
                file_path = self.datatransformationconfig.processed_file_path,
                obj = processor_obj
            )

            logging.info(f"Saving of file_path - {self.datatransformationconfig.processed_file_path} and processor object done!!")
            logging.info(f"{train_transformed_arr[:5]}")

            logging.info("Saving done!!! Data Transformation done!!!")

            return (
                train_transformed_arr,
                test_transformed_arr,
                self.datatransformationconfig.processed_file_path
            )
        
        except Exception as e:
            logging.info("Exception has occured while data transformation")
            raise CustomException(e, sys)
