import joblib
import pandas as pd

from walmart_utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET,
    calculate_metrics,
    get_logger,
    load_processed_split,
    save_json,
)


def main():
    logger = get_logger("evaluate", "evaluate.log")
    logger.info("Starting model evaluation")
    X_test, y_test = load_processed_split("test.csv")
    model = joblib.load(MODELS_DIR / "random_forest.joblib")
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"Actual": y_test.to_numpy(), "Prediction": predictions})
    results["Residual"] = results["Actual"] - results["Prediction"]
    results["Absolute_Error"] = results["Residual"].abs()
    results.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    results.sort_values("Absolute_Error", ascending=False).to_csv(
        OUTPUTS_DIR / "error_analysis.csv", index=False
    )
    save_json("artifacts/lab3_metrics.json", metrics)
    print("\n--- Model Evaluation Report ---")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2  : {metrics['R2']:.4f}")
    print("Evaluation complete; predictions and error analysis saved.")
    logger.info("Evaluation complete: %s", metrics)


if __name__ == "__main__":
    main()
