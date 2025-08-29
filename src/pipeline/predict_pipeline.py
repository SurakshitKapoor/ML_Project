
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
    
    def predict(self, features):
        try:

            logging.info("Starting to Predict the result from trained model.")
            if isinstance(features, dict):
                features = pd.DataFrame(features)

            preprocessor_pkl = os.path.join("artifacts", "preprocessor.pkl")
            model_pkl = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_pkl)
            model = load_object(model_pkl)

            scaled_features = preprocessor.transform(features)
            pred = model.predict(scaled_features)
            
            logging.info("Model Predicted the results!!!")
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)

# obj = PredictPipeline()



class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dict(self):
        return {
            "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score]
        }
