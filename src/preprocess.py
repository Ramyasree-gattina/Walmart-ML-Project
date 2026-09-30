from walmart_utils import PROCESSED_DIR, TARGET, get_logger, load_raw_data, save_json, split_raw_data


def main():
    logger = get_logger("preprocess", "preprocess.log")
    logger.info("Starting preprocessing pipeline")
    data = load_raw_data()
    X_train, X_test, y_train, y_test = split_raw_data(data)
    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    save_json(
        PROCESSED_DIR / "split_summary.json",
        {
            "records": len(data),
            "train_records": len(train),
            "test_records": len(test),
            "features": list(X_train.columns),
        },
    )
    logger.info("Saved %s training and %s test records", len(train), len(test))


if __name__ == "__main__":
    main()
