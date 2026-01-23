import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class DataTransformation:

    def get_data_transformer_object(self, df):
        try:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

            if "diagnosed_diabetes" in numerical_cols:
                numerical_cols.remove("diagnosed_diabetes")

            logger.info(f"Categorical columns: {categorical_cols}")
            logger.info(f"Numerical columns: {numerical_cols}")

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Data Transformation started")

            # 🔥 READ CSVs HERE (critical)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape: {test_df.shape}")

            target_col = "diagnosed_diabetes"

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            preprocessor = self.get_data_transformer_object(X_train)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(
                file_path=os.path.join("artifacts", "preprocessor.pkl"),
                obj=preprocessor
            )

            logger.info("Data Transformation completed successfully")

            return X_train_arr, X_test_arr, y_train, y_test

        except Exception as e:
            logger.exception("Error occurred during data transformation")
            raise CustomException(e, sys)

