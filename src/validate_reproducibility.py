import numpy as np

from walmart_utils import ARTIFACTS_DIR, create_model, get_logger, load_processed_split, save_json


def main():
    logger = get_logger("validate_reproducibility", "reproducibility.log")
    X_train, y_train = load_processed_split("train.csv")
    X_test, _ = load_processed_split("test.csv")
    predictions = []
    for _ in range(2):
        model = create_model()
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))
    identical = bool(np.array_equal(predictions[0], predictions[1]))
    report = {
        "runs": 2,
        "predictions_identical": identical,
        "max_absolute_prediction_difference": float(
            np.max(np.abs(predictions[0] - predictions[1]))
        ),
    }
    save_json(ARTIFACTS_DIR / "reproducibility_report.json", report)
    if not identical:
        raise RuntimeError("Repeated training produced different predictions.")
    logger.info("Reproducibility validation PASSED; predictions are identical")


if __name__ == "__main__":
    main()
