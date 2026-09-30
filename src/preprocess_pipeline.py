import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from walmart_utils import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    MODELS_DIR,
    PROCESSED_DIR,
    TARGET,
    engineer_features,
    get_logger,
    load_raw_data,
    save_json,
    validate_raw_data,
)


def main():
    logger = get_logger("preprocess_pipeline", "preprocess_pipeline.log")
    data = load_raw_data()
    validate_raw_data(data)
    raw_features = data.drop(columns=[TARGET])
    target = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        raw_features, target, test_size=0.2, random_state=42
    )
    preprocessor = Pipeline(
        steps=[
            ("date_features", FunctionTransformer(engineer_features, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    logger.info("Fitting preprocessing pipeline on training data")
    train_values = preprocessor.fit_transform(X_train)
    test_values = preprocessor.transform(X_test)
    train = pd.DataFrame(train_values, columns=FEATURE_COLUMNS)
    train[TARGET] = y_train.to_numpy()
    test = pd.DataFrame(test_values, columns=FEATURE_COLUMNS)
    test[TARGET] = y_test.to_numpy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "pipeline_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "pipeline_test.csv", index=False)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    report = {
        "train_records": len(train),
        "test_records": len(test),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "train_missing_values": int(np.isnan(train[FEATURE_COLUMNS].to_numpy()).sum()),
        "test_missing_values": int(np.isnan(test[FEATURE_COLUMNS].to_numpy()).sum()),
    }
    save_json(ARTIFACTS_DIR / "preprocessing_summary_report.json", report)
    logger.info("Preprocessing complete; saved fitted pipeline and transformed splits")


if __name__ == "__main__":
    main()
