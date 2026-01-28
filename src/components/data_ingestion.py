import logging
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from src.exception import CustomException
from src.logger import logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.components.model_tuning import ModelTuning


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")
        try:
            df = pd.read_csv("data/raw/train.csv")
            logging.info("Dataset read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved successfully")
            logging.info("Splitting data into train and test sets")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.exception("Error occurred during data ingestion")
            raise CustomException(e, sys)


from src.components.model_tuning import ModelTuning

if __name__ == "__main__":

    # 1) Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2) Transformation
    data_transformation = DataTransformation()
    X_train_arr, X_test_arr, y_train, y_test = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3) Training (optional)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(np.c_[X_train_arr, y_train], np.c_[X_test_arr, y_test])

    # 4) Tuning
    tuner = ModelTuning()
    print(tuner.tune_model(X_train_arr, y_train, X_test_arr, y_test))
    
    
    
