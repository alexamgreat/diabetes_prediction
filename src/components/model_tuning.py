import os, sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


class ModelTuning:

    def __init__(self):
        pass

    def tune_model(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("Model Tuning started")

            # ---- Random Forest ----
            rf = RandomForestClassifier(
                random_state=42,
                n_jobs=1  # prevent nested parallelism
            )

            rf_params = {
                "n_estimators": [100],
                "max_depth": [None, 20],
                "min_samples_split": [2],
                "min_samples_leaf": [1]
            }

            rf_grid = GridSearchCV(
                estimator=rf,
                param_grid=rf_params,
                scoring="f1",
                cv=3,
                n_jobs=1,   # 🔥 critical fix
                verbose=1
            )

            rf_grid.fit(X_train, y_train)
            rf_best = rf_grid.best_estimator_
            rf_f1 = f1_score(y_test, rf_best.predict(X_test))

            # ---- XGBoost ----
            xgb = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                tree_method="hist"  # 🔥 memory efficient
            )

            xgb_params = {
                "n_estimators": [100],
                "max_depth": [3, 6],
                "learning_rate": [0.1],
                "subsample": [0.8]
            }

            xgb_grid = GridSearchCV(
                estimator=xgb,
                param_grid=xgb_params,
                scoring="f1",
                cv=3,
                n_jobs=1,   # 🔥 critical fix
                verbose=1
            )

            xgb_grid.fit(X_train, y_train)
            xgb_best = xgb_grid.best_estimator_
            xgb_f1 = f1_score(y_test, xgb_best.predict(X_test))

            # ---- Select Best Model ----
            best_model = rf_best if rf_f1 > xgb_f1 else xgb_best
            best_f1 = max(rf_f1, xgb_f1)

            save_object(
                os.path.join("artifacts", "model.pkl"),
                best_model
            )

            logger.info(f"Model tuning completed. Best F1 score: {best_f1}")

            return {
                "rf_f1": rf_f1,
                "xgb_f1": xgb_f1,
                "best_f1": best_f1
            }

        except Exception as e:
            logger.exception("Error during model tuning")
            raise CustomException(e, sys)
