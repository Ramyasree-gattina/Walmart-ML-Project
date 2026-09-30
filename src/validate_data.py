from walmart_utils import get_logger, load_raw_data, validate_raw_data


def main():
    logger = get_logger("validate_data", "validate_data.log")
    data = load_raw_data()
    logger.info("Validating schema (records: %s)", len(data))
    validate_raw_data(data)
    logger.info("Schema validation PASSED; dataset is clean")


if __name__ == "__main__":
    main()
