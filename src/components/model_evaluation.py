import numpy as np
import pickle
import os
import sys
import mlflow
import mlflow.sklearn

from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException


class ModelEvaluator:

    def __init__(self, experiment_name="ML Project Evaluation"):
        self.experiment_name = experiment_name

    def eval_metrics(self, actual, pred):
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return mse, mae, r2

    def initiate_model_evaluation(self, train_arr, test_arr):
        try:
            logging.info("Splitting test data into X_test and y_test")
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            logging.info("Starting MLflow run...")

            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run():
                logging.info("Generating predictions...")
                predictions = model.predict(X_test)

                mse, mae, r2 = self.eval_metrics(y_test, predictions)

                logging.info(f"Evaluation Metrics -> MSE: {mse}, MAE: {mae}, R2: {r2}")

                # Log parameters if model has them (like sklearn models)
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Log metrics
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_url_type_store != "file":
                    # Register the model
                    mlflow.sklearn.log_model(model, "model", registered_model_name="StudentPerformanceModel")
                else:
                    mlflow.sklearn.log_model(model, "model")

                logging.info("Model evaluation completed and logged to MLflow")

        except Exception as e:
            raise CustomException(e, sys)
