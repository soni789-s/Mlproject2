import os
import sys
import pandas as pd
from utility.exception import CustomException
from utility.loggers import logger
from sklearn.model_selection import train_test_split
from src.data_preprocessing import Data_Preprocessing
from sklearn.preprocessing import LabelEncoder
from src.model_training import Model_Training
from src.model_evaluation import Model_Evaluation


class Data_Ingestion:
    def __init__(self, arifacts_dict):
        self.artifacts_dict = arifacts_dict
        self.raw_df = os.path.join(self.artifacts_dict, "raw.csv")
        self.train_df = os.path.join(self.artifacts_dict, "train.csv")
        self.test_df = os.path.join(self.artifacts_dict, "test.csv")
        os.makedirs(self.artifacts_dict, exist_ok=True)

    def initiate_data_ingestion(self):
        try:
            # Load the dataset
            df = pd.read_csv(r"C:\Users\udaya\OneDrive\Desktop\Mlproject2\notebook\spam.csv")
            logger.info("Dataset loaded successfully")
            
            # Rename columns
            df.rename(columns={"Message": "text", "Category": "target"}, inplace=True)

            # Check for missing values
            if df['text'].isnull().sum() > 0 or df['target'].isnull().sum() > 0:
                logger.warning("Missing values found in the dataset")

            # Remove duplicates
            if df.duplicated().sum():
                df.drop_duplicates(inplace=True)

            # Encode the target variable
            l = LabelEncoder()
            df['target'] = l.fit_transform(df['target'])
            
            # Save the cleaned data
            df.to_csv(self.raw_df, index=False, header=True)
            logger.info("Raw data saved to raw.csv")
            
            # Split data into training and testing sets
            logger.info("Splitting the data into training and testing sets")
            # 0.2 represents 20% of data as test set, random_state=42 ensures reproducibility of the split. 80% of data is used for training. 20% is used for testing. 42 is a random seed. 0.2 is a common split ratio. 80% is 0.8 * 100% = 80% of the data. 20% is 0.2 * 100% = 20% of the data. 42 is a random seed to ensure reproducibility. 0.2 is a common split ratio.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=2)
            logger.info("spliting is done")
            train_set.to_csv(self.train_df, index=False, header=True)
            logger.info("Training data saved to train.csv")
            test_set.to_csv(self.test_df, index=False, header=True)
            logger.info("Testing data saved to test.csv")
            logger.info("Training and test data saved to train.csv and test.csv")

            return train_set, test_set

        except Exception as e:
            logger.error("Error occurred while loading data")
            print(CustomException(e, sys))


if __name__ == '__main__':
    data_ingestion = Data_Ingestion("artifacts")
    train_df, test_df = data_ingestion.initiate_data_ingestion()

    data_preprocessing = Data_Preprocessing(train_df, test_df)
    train_arr,test_arr =data_preprocessing.initiate_data_preprocessing()

    model_training = Model_Training(train_arr, test_arr)
    models_dict,X_train,y_train,X_test,y_test = model_training.initiate_model_training()

    model_evaluation = Model_Evaluation(models_dict, X_train, y_train, X_test, y_test)
    model_evaluation.initiate_model_evaluation()
