import os
import sys
import numpy as np
from sklearn.svm import SVC
from utility.exception import CustomException
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from xgboost import XGBClassifier
from utility.loggers import logger

class Model_Training:
    def __init__(self,train_arr,test_arr):
        self.train_arr = train_arr
        self.test_arr = test_arr
    def initiate_model_training(self):
        try:
            X_train, y_train = self.train_arr[:, :-1], self.train_arr[:, -1]
            X_test, y_test = self.test_arr[:, :-1], self.test_arr[:, -1]

            ada = AdaBoostClassifier(n_estimators=50,random_state=2)
            bagging = BaggingClassifier(n_estimators=50,random_state=2)
            extra = ExtraTreesClassifier(n_estimators=50,random_state=2)
            gradient = GradientBoostingClassifier(n_estimators=50,random_state=2)
            random = RandomForestClassifier(n_estimators=50,random_state=2)
            svc = SVC(kernel='sigmoid', gamma=1.0,random_state=2)
            decision = DecisionTreeClassifier(random_state=2,max_depth=5)
            logistic = LogisticRegression(solver='liblinear', penalty='l1')
            multi = MultinomialNB()
            gaussian = GaussianNB()
            bernoulli = BernoulliNB()
            knc = KNeighborsClassifier()
            xgb = XGBClassifier(n_estimators=50,random_state=2)

            logger.info("models trained successfully")

            models_dict  = {"ADA_BOOST":ada,
                    "BAGGIMG":bagging,
                    "EXTRA_TREES":extra,
                    "GRADIENT_BOOST":gradient,
                    "RANDOM_FOREST":random,
                    "SVC":svc,
                    "DECISION_TREE":decision,
                    "LOGISTIC_REGREESION":logistic,
                    "MULTINOMINAL":multi,
                    "GAUSSIAN":gaussian,
                    "BERNOULLI":bernoulli,
                    "KNN":knc,
                    "XGB":xgb}
            
            return models_dict,X_train,y_train,X_test,y_test
        except Exception as e:
            logger.error("Error in training models")
            print(CustomException(e,sys))