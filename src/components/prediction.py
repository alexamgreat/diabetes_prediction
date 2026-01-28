import pandas as pd
from src.utils import load_object

def run_prediction(test_path="data/raw/test.csv", submission_path="submission.csv"):
    try:
        # 🔹 Load test data
        df_test = pd.read_csv(test_path)

        if "id" not in df_test.columns:
            raise ValueError(" 'id' column is required in test data")

        ids = df_test["id"]

        #  Load artifacts
        preprocessor = load_object("artifacts/preprocessor.pkl")
        model = load_object("artifacts/model.pkl")

        #  Make test data match preprocessor
        expected_features = preprocessor.feature_names_in_

        # Add missing columns with default 
        for col in expected_features:
            if col not in df_test.columns:
                df_test[col] = 0

        # Reorder columns exactly as expected
        X_test = df_test[expected_features]

        #  Transform & predict
        X_test_transformed = preprocessor.transform(X_test)
        y_pred_prob = model.predict_proba(X_test_transformed)[:, 1]

        #  Create submission
        submission = pd.DataFrame({
            "id": ids,
            "diagnosed_diabetes": y_pred_prob
        })

        submission.to_csv(submission_path, index=False)
        print(f"✅ submission saved successfully at '{submission_path}'")

    except Exception as e:
        print(" Prediction failed:", str(e))


if __name__ == "__main__":
    run_prediction()