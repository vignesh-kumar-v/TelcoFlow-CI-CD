.PHONY: help install validate sql-features train train-quick tune score score-drift \
        ab-test pipeline test unit-test integration-test api clean \
        docker-build docker-run compose-up compose-down \
        mlflow-ui bq-load bq-features bq-analytics bq-predictions \
        airflow-install airflow-run k8s-deploy k8s-status k8s-delete

# Prefer the project venv when present, so `make` works without activating it.
PYTHON ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python)
AIRFLOW_VENV ?= .venv-airflow
AIRFLOW_CONSTRAINTS ?= https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.11.txt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

# --- Data ---------------------------------------------------------------
validate: ## Clean and validate the raw data with pandas
	$(PYTHON) -m src.telco_churn.validate_and_clean

sql-features: ## Ingest to SQL and build engineered features (DB_URL selects the backend)
	$(PYTHON) -m src.telco_churn.sql_features

# --- Training -----------------------------------------------------------
train: ## Train all models with Optuna tuning
	$(PYTHON) -m src.telco_churn.train

train-quick: ## Train all models without Optuna tuning
	$(PYTHON) -m src.telco_churn.train --skip-tuning

tune: ## Train with 100 Optuna trials
	$(PYTHON) -m src.telco_churn.train --n-trials 100

# --- Scoring and analysis ----------------------------------------------
score: ## Score the held-out batch and write a drift report
	$(PYTHON) -m src.telco_churn.batch_score

score-drift: ## Score a deliberately perturbed batch to demonstrate drift detection
	$(PYTHON) -m src.telco_churn.batch_score --simulate-drift

ab-test: ## Simulate a retention campaign and test for significance
	$(PYTHON) -m src.telco_churn.ab_test

pipeline: sql-features train-quick score ab-test ## Run the whole pipeline end to end

# --- Tests --------------------------------------------------------------
test: ## Run every test
	$(PYTHON) -m pytest

unit-test: ## Run only tests that need no pipeline artifacts
	$(PYTHON) -m pytest -m "not integration and not gcp" -v

integration-test: ## Run only tests that need a completed pipeline run
	$(PYTHON) -m pytest -m integration -v

# --- Serving ------------------------------------------------------------
api: ## Start the FastAPI server locally
	$(PYTHON) -m src.telco_churn.api

# --- Experiment tracking ------------------------------------------------
# macOS binds port 5000 to the AirPlay Receiver, so make it overridable:
#   make mlflow-ui MLFLOW_PORT=5001
MLFLOW_PORT ?= 5000

mlflow-ui: ## Browse runs and the model registry (MLFLOW_PORT=5001 to override)
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

# --- Docker -------------------------------------------------------------
# Bump VERSION when the image changes and you want Kubernetes to pick it up:
# the cluster keeps its own image store, so reusing a tag leaves stale pods.
# Must match the image tag in k8s/*.yaml (tests/test_k8s_manifests.py enforces it).
VERSION ?= 0.2.0

docker-build: ## Build the Docker image (tags $(VERSION) and latest)
	docker build -t telco-churn-mlops:$(VERSION) -t telco-churn-mlops:latest .

docker-run: ## Run the API in Docker
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/home/mluser/app/data \
		-v $(PWD)/outputs:/home/mluser/app/outputs \
		-v $(PWD)/artifacts:/home/mluser/app/artifacts \
		telco-churn-mlops python -m src.telco_churn.api

compose-up: ## Start Postgres and the MLflow server
	docker compose up -d

compose-down: ## Stop the Compose stack
	docker compose down

# --- BigQuery -----------------------------------------------------------
# Falls back to the active gcloud project, so an explicit export is optional.
# Needs application-default credentials:
#   gcloud auth application-default login
GCP_PROJECT ?= $(shell gcloud config get-value project 2>/dev/null)
export GCP_PROJECT

bq-load: ## Load the cleaned data into BigQuery
	$(PYTHON) -m src.telco_churn.bigquery_loader load

bq-features: ## Query engineered features back out of BigQuery
	$(PYTHON) -m src.telco_churn.bigquery_loader features

bq-analytics: ## Compute churn by segment in BigQuery
	$(PYTHON) -m src.telco_churn.bigquery_loader analytics

bq-predictions: ## Publish the latest predictions to BigQuery
	$(PYTHON) -m src.telco_churn.bigquery_loader upload-predictions

bq-all: bq-load bq-analytics bq-predictions ## Load, analyse and publish in one go

# --- Airflow (separate environment) -------------------------------------
airflow-install: ## Create an isolated venv and install Airflow with official constraints
	python -m venv $(AIRFLOW_VENV)
	$(AIRFLOW_VENV)/bin/pip install --upgrade pip
	$(AIRFLOW_VENV)/bin/pip install -r requirements-airflow.txt --constraint "$(AIRFLOW_CONSTRAINTS)"

airflow-run: ## Start Airflow locally (UI at http://localhost:8080)
	AIRFLOW_HOME=$(PWD)/airflow \
	AIRFLOW__CORE__DAGS_FOLDER=$(PWD)/airflow/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	$(AIRFLOW_VENV)/bin/airflow standalone

# --- Kubernetes ---------------------------------------------------------
k8s-deploy: ## Apply all manifests to the current cluster
	kubectl apply -f k8s/base.yaml
	kubectl apply -f k8s/jobs.yaml
	kubectl apply -f k8s/api-deployment.yaml

k8s-status: ## Show workloads in the telco-churn namespace
	kubectl get pods,svc,jobs,cronjobs,pvc -n telco-churn

k8s-delete: ## Tear down the namespace and everything in it
	kubectl delete namespace telco-churn --ignore-not-found

# --- Housekeeping -------------------------------------------------------
clean: ## Remove bytecode and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
