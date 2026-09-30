import joblib

from walmart_utils import MODELS_DIR, create_model, get_logger, load_processed_split


def main():
    logger = get_logger("train", "train.log")
    logger.info("Starting Random Forest training")
    X_train, y_train = load_processed_split("train.csv")
    model = create_model()
    model.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(model, model_path)
    logger.info("Model trained and saved to %s", model_path)


if __name__ == "__main__":
    main()
