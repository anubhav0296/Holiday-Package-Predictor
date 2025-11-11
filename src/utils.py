import os
import sys
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score, recall_score
# from sklearn.model_selection import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        logging.info(f"Saving object has started. File path is {file_path}")
        dir_path = os.path.dirname(file_path)

        logging.info("Making folder where the file will be saved")
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Saving object is successfully complete")

    except Exception as e:
        logging.info("Exception occurerd while saving object")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():    

            para = params.get(model_name, {})

            if para:
                # gs = GridSearchCV(model, para, cv=5)
                gs = RandomizedSearchCV(model, para, n_iter=20, cv=3, random_state=42, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_pred_test = model.predict(X_test)

            acc_score = accuracy_score(y_test, y_pred_test)
            prec_score = precision_score(y_test, y_pred_test)
            recal_score = recall_score(y_test, y_pred_test)
            f1score = f1_score(y_test, y_pred_test)

            report[model_name] = {"Accuracy": acc_score, "Precision": prec_score, "Recall": recal_score, "F1": f1score}

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)