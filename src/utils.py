import os
import sys
import dill

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logger.exception("Error occurred while saving object")
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        logger.exception("Error occurred while loading object")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    from sklearn.metrics import accuracy_score

    # Ensure y is binary
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Ensure only binary classification
    if len(set(y_train)) > 2:
        raise Exception("Target variable is not binary. Check y_train values.")

    model_report = {}
    try:
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            model_report[model_name] = score
            logger.info(f"{model_name} evaluated with accuracy: {score}")

        return model_report

    except Exception as e:
        logger.exception("Error occurred while evaluating models")
        raise CustomException(e, sys)