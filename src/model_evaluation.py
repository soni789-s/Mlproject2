import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from utility.exception import CustomException
from utility.loggers import logger
from utility.utils import save_object
from sklearn.metrics import confusion_matrix

class Model_Evaluation:
    def __init__(self,models_dict,X_train,y_train,X_test,y_test):
        self.models_dict = models_dict
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
    
    def initiate_model_evaluation(self):
        try:
            logger.info("Starting model evaluation")
            results = []
            for name, model in self.models_dict.items():
                print(name,model)
                models = model.fit(self.X_train, self.y_train)
                self.y_pred_train = models.predict(self.X_train)
                accuracy_train = accuracy_score(self.y_train, self.y_pred_train)
                self.y_pred = models.predict(self.X_test)
                accuracy_test= accuracy_score(self.y_test, self.y_pred)
                cross_validation = cross_val_score(models, self.X_train, self.y_train, cv=10).mean()
                precision = precision_score(self.y_test, self.y_pred)
                logger.info(f"Model Evaluation for {name} --> accuracy for training {accuracy_train}  and for test {accuracy_test} and cross validation {cross_validation} and precision {precision}")
                model_eval_dict = {
                    'model':name,
                    'accuracy': accuracy_test,
                    'cross_validation': cross_validation,
                    'precision': precision
                }
                results.append(model_eval_dict)
            logger.info("successfully model_evaluation_dictionary results list created")
            performance_df = pd.DataFrame(results)
            logger.info("Performance DataFrame created successfully")
            sorted_df = performance_df.sort_values(by=['accuracy','precision'],ascending=False)

            best_model = sorted_df.iloc[0,0]
            best_model_name = self.models_dict[best_model] 
            save_object("artifacts/model.pkl", best_model_name)


        except Exception as e:
            logger.error("Error occurred while model evaluation")
            print(CustomException(e,sys))