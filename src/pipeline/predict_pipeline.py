
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        print("__init__ of PredictPipeline")
        pass
    
    def __new__(self):
        print("__new__ of PredictPipeline")
        pass

    def predict(self, features):
        try:
            preprocessor_pkl = os.path.join("artifacts", "preprocessor.pkl")
            model_pkl = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_pkl)
            model = load_object(model_pkl)

            scaled_features = preprocessor.transform(features)
            pred = model.predict(scaled_features)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

# obj = PredictPipeline()



class CustomData:
    def __init__(self):
        pass

    def get_data_as_dataframe(self):
        pass
