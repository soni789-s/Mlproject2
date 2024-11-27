import os
import sys
import pandas as pd
from utility.utils import load_object
from utility.exception import CustomException
from scipy.sparse import csr_matrix
from utility.loggers import logger

class Predict_Pipeline:
    def __init__(self,message):
        self.message = message
    
    def customdata(self):
        try:
            self.dict = {"text": [self.message]}
            predict_df = pd.DataFrame(self.dict)
            return predict_df
        except Exception as e:
            logger.error("Error is raised when custom data is used")
            print(CustomException(e,sys))
    
    def predict(self,dataframe):
        try:
            preprocessor_path = os.path.join("artifacts","preprocessing.pkl")
            model_path = os.path.join("artifacts","model.pkl")
            logger.info("loaded preprocessing and model paths")
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data = preprocessor.transform(dataframe)
            input_feature = csr_matrix(data).toarray()
            prediction = model.predict(input_feature)
            logger.info("prediction is done")

            if prediction[0] == 0.:
                return "NOT SPAM"
            else:
                return "SPAM"
        except Exception as e:
            logger.error("Error occurred while predicting")
            print(CustomException(e,sys))
