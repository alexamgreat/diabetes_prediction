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
