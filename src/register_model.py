import json

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

from train_mlflow import tracking_uri
from walmart_utils import ARTIFACTS_DIR, get_logger, load_processed_split, save_json


REGISTERED_MODEL_NAME = "WalmartWeeklySales"


def main():
    logger = get_logger("register_model", "register_model.log")
    with (ARTIFACTS_DIR / "lab4_tracking_report.json").open(encoding="utf-8") as file:
        run_report = json.load(file)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_registry_uri(tracking_uri())
    X_train, y_train = load_processed_split("train.csv")
    X_test, y_test = load_processed_split("test.csv")
    model_uri = f"runs:/{run_report['run_id']}/model"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(X_test)
    candidate_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    baseline_prediction = np.full(len(y_test), float(y_train.mean()))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_prediction)))
    promoted = bool(np.isfinite(candidate_rmse) and candidate_rmse < baseline_rmse)
    report = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "candidate_rmse": candidate_rmse,
        "mean_baseline_rmse": baseline_rmse,
        "quality_gate_passed": promoted,
        "alias": None,
    }
    if not promoted:
        save_json(ARTIFACTS_DIR / "registry_report.json", report)
        raise RuntimeError("Model did not beat the mean-prediction baseline; not promoted.")
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = MlflowClient(tracking_uri=tracking_uri(), registry_uri=tracking_uri())
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", registered.version)
    report["version"] = registered.version
    report["alias"] = "champion"
    report["model_uri"] = model_uri
    save_json(ARTIFACTS_DIR / "registry_report.json", report)
    logger.info(
        "Registered version %s and assigned the champion alias (RMSE %.2f < %.2f)",
        registered.version,
        candidate_rmse,
        baseline_rmse,
    )


if __name__ == "__main__":
    main()
