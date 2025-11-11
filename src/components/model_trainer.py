from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from utils import evaluate_model, save_object

from src.logger import logging
from src.exception import CustomException
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.components.data_transformation import DataTransformation


@dataclass 
class ModelTrainerConfig:
    model_file_path:str = os.path.join("artifacts", "model.pkl")
    # transformed_test_data_path:str = os.path.join("artifacts", "transformed_test_data.pxl")

class ModelTrainer:
    def __init__(self):
        self.modelconfig = ModelTrainerConfig()

    def get_best_model(self, train_arr, test_arr):
        try:
            X_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            X_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]
            
            models={
                "Logistic Regression":LogisticRegression(),
                "Decision Tree":DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "Gradient Boost":GradientBoostingClassifier()
            }

            params = {
                "Logisitic Regression" : {
                    "penalty": ["l1", "l2", "elasticnet", "none"],  # type of regularization
                    "C": [0.01, 0.1, 1, 10, 100],                 # inverse regularization strength
                    "solver": ["liblinear", "saga"],     # depends on penalty type
                    "max_iter": [100, 200, 500]
                },

                "Decision Tree" : {
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 3, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10, 20],
                    "min_samples_leaf": [1, 2, 5, 10],
                    "max_features": [None, "sqrt", "log2"]
                },

                ## Hyperparameter for Random Forest Classifier
                "Random Forest" : {
                    "n_estimators": [100, 200, 300, 500, 800, 1000],
                    "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10, 15],
                    "min_samples_leaf": [1, 2, 5, 10],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                },

                "Gradient Boost" : {
                    "n_estimators": [100, 200, 300, 500],      # number of boosting stages
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],   # step size shrinkage
                    "max_depth": [3, 4, 5, 6, 8],              # depth of individual trees
                    "min_samples_split": [2, 5, 10, 20],       # min samples to split an internal node
                    "min_samples_leaf": [1, 2, 5, 10],         # min samples at a leaf node
                    "subsample": [0.6, 0.8, 1.0],              # fraction of samples used per tree
                    "max_features": ["sqrt", "log2", None]     # number of features to consider when looking for best split
                }
            }

            # This function takes the inputs, runs a loop to fit the data in each model 
            # the returns as reportof each model (Dictionary) 
            model_eval = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            results_df = pd.DataFrame(model_eval).T

            logging.info(results_df.sort_values(by="F1", ascending=False))

            best_model_name = results_df['F1'].idxmax()
            best_model = models[best_model_name]
            best_score = results_df.loc[best_model_name, 'F1']

            logging.info(f"✅ Best model: {best_model_name} (F1 = {best_score:.4f})")

            save_object(self.modelconfig.model_file_path, best_model)

            return best_model_name, best_model, best_score

        except Exception as e:
            raise CustomException(e, sys)


