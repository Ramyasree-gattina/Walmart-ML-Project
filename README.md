# Walmart Weekly Sales Prediction

An end-to-end regression project that predicts `Weekly_Sales` from Walmart store, date, holiday, and economic data.

## Project layout

```text
data/raw/          Original dataset
data/processed/    Reproducible train/test datasets
notebooks/         Exploratory analysis
src/               Preprocessing, training, evaluation, tracking, and validation
pipelines/         Lab 3-6 and all-in-one runners
models/            Serialized model and preprocessing pipeline
outputs/           Predictions and error analysis
logs/              Script logs
artifacts/         Metrics, validation reports, and MLflow database
```

## Setup

From the project root, create or activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The source dataset belongs at `data/raw/Walmart.csv`.

## Run the labs

```powershell
python .\pipelines\run_lab3_baseline.py
python .\pipelines\run_lab4_tracking.py
python .\pipelines\run_lab5_pipeline.py
python .\pipelines\run_lab6_registry.py
```

Run every stage in order with `python .\pipelines\run_all.py`. MLflow tracking and the model registry are stored locally in `artifacts/mlflow.db`; start the UI with `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`.

## GitHub

Create an empty GitHub repository, then from this folder run `git init`, `git add .`, and `git commit -m "Build Walmart sales ML pipeline"`. Add the repository URL as `origin`, then push the current branch. The virtual environment, database, trained models, and generated outputs are excluded by `.gitignore`; the raw CSV and notebook remain part of the project.
