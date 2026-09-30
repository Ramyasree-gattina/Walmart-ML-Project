import mlflow
import mlflow.sklearn

from walmart_utils import (
    ARTIFACTS_DIR,
    calculate_metrics,
    create_model,
    get_logger,
    load_processed_split,
    save_json,
)


def tracking_uri():
    return f"sqlite:///{(ARTIFACTS_DIR / 'mlflow.db').as_posix()}"


def main():
    logger = get_logger("train_mlflow", "train_mlflow.log")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("Walmart Weekly Sales")
    X_train, y_train = load_processed_split("train.csv")
    X_test, y_test = load_processed_split("test.csv")
    model = create_model()
    with mlflow.start_run(run_name="RandomForestRegressor") as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)
        mlflow.log_params(
            {"model": "RandomForestRegressor", "n_estimators": 100, "random_state": 42}
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, artifact_path="model", serialization_format="cloudpickle"
        )
        report = {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "tracking_uri": tracking_uri(),
        }
    save_json(ARTIFACTS_DIR / "lab4_tracking_report.json", report)
    print(f"MLflow run tracked: RMSE = {metrics['RMSE']:.2f} | R2 = {metrics['R2']:.4f}")
    logger.info("Run %s successfully tracked", run.info.run_id)


if __name__ == "__main__":
    main()
