import joblib
import numpy as np
import pandas as pd

from walmart_utils import FEATURE_COLUMNS, MODELS_DIR, PROCESSED_DIR, TARGET, get_logger


def main():
    logger = get_logger("validate_outputs", "validate_outputs.log")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    if preprocessor is None:
        raise ValueError("Saved preprocessor is empty.")
    for split in ("pipeline_train.csv", "pipeline_test.csv"):
        data = pd.read_csv(PROCESSED_DIR / split)
        if list(data.columns) != FEATURE_COLUMNS + [TARGET]:
            raise ValueError(f"Unexpected feature columns in {split}.")
        values = data[FEATURE_COLUMNS].to_numpy()
        if not np.isfinite(values).all() or data[TARGET].isna().any():
            raise ValueError(f"Non-finite or missing values found in {split}.")
    logger.info("Output validation PASSED; features are clean and dimensionally consistent")


if __name__ == "__main__":
    main()
