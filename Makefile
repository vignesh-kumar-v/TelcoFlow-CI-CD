.PHONY: install validate train score test clean

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