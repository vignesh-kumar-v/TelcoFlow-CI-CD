.PHONY: install validate train score test clean api docker-build docker-run unit-test

install:
	pip install -r requirements.txt

validate:
	python -m src.telco_churn.validate_and_clean

train:
	python -m src.telco_churn.train

score:
	python -m src.telco_churn.batch_score

test:
	PYTHONPATH=. pytest tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

# API commands
api:
	python -m src.telco_churn.api

docker-build:
	docker build -t telco-churn-mlops .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $(PWD)/data:/home/mluser/app/data \
		-v $(PWD)/outputs:/home/mluser/app/outputs \
		-v $(PWD)/artifacts:/home/mluser/app/artifacts \
		telco-churn-mlops python -m src.telco_churn.api

unit-test:
	pytest tests/test_unit.py -v