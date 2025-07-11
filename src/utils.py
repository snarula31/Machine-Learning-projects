import os,sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill


def save_object(file_path, obj):
    """
    This function saves the object to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    This function evaluates the given models and returns a report of their performance.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            models = list(models.values())[i]
            models.fit(X_train, y_train)
            y_train_pred = models.predict(X_train)
            y_test_pred = models.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys) from e