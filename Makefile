.PHONY: install data train evaluate api dashboard docker-build clean

# ── Variables ──────────────────────────────────────────────────────────
PYTHON ?= python

# ── Targets ────────────────────────────────────────────────────────────

## Install all dependencies
install:
	pip install -r requirements.txt

## Generate synthetic transaction data
data:
	$(PYTHON) -m src.data.generate_data

## Run feature engineering and train all models
train: 
	$(PYTHON) -m src.features.feature_engineering
	$(PYTHON) -m src.models.train

## Evaluate best model and generate plots
evaluate:
	$(PYTHON) -m src.models.evaluate

## Start FastAPI server (development)
api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

## Start Streamlit dashboard (development)
dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

## Build Docker images
docker-build:
	docker-compose build

## Run Docker services
docker-up:
	docker-compose up -d

## Stop Docker services
docker-down:
	docker-compose down

## Remove generated data, models, and results
clean:
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.csv data/processed/*.pkl data/processed/*.json
	rm -rf models/*.joblib models/*.pkl
	rm -rf results/figures/*.png results/metrics.json

## Run full pipeline: data → train → evaluate
all: data train evaluate
