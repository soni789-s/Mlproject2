import sys
import string
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from utility.exception import CustomException
from utility.utils import save_object
from utility.loggers import logger
from scipy.sparse import csr_matrix


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        logger.info("I was able to fit")
        return self
        

    def transform(self, X):
        try:
            processed = []
            logger.info("I am bout to entry into for loop...")
            for text in X:
                tokens = nltk.word_tokenize(text)
                tokens = [self.ps.stem(word) for word in tokens if word not in self.stop_words and word not in string.punctuation]
                processed.append(' '.join(tokens))
            logger.info("Text preprocessing done successfully")
            return processed 
        except Exception as e:
            logger.error("Error occurred while preprocessing text")
            print(CustomException(e,sys))

class Data_Preprocessing:
    def __init__(self,train_df,test_df):
        self.train_df = train_df
        self.test_df = test_df
    
    def initiate_data_preprocessing(self):
        try:
            pipeline = Pipeline([("text_preprocessing",TextPreprocessor()),('tfidf', TfidfVectorizer(max_features=3000))])
            logger.info("Text preprocessing pipeline is initiated successfully")
            input_feature_train_arr=pipeline.fit_transform(self.train_df['text'])
            input_feature_test_arr=pipeline.transform(self.test_df['text'])
            input_feature_train_ar = csr_matrix(input_feature_train_arr).toarray()
            train_arr = np.c_[input_feature_train_ar,np.array(self.train_df['target'])]
            input_feature_test_ar = csr_matrix(input_feature_test_arr).toarray()
            test_arr = np.c_[input_feature_test_ar,np.array(self.test_df['target'])]
            logger.info("got the train and test array")
            save_object(r"artifacts\preprocessing.pkl",pipeline)
            logger.info("pickle file is saved successfully")
            return train_arr, test_arr
        except Exception as e:
            logger.error("Error occurred while preprocessing data")
            print(CustomException(e,sys))

 


