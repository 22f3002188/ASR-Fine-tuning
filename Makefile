.PHONY: data train eval infer test lint

data:
	python scripts/prepare_data.py

train:
	python scripts/train.py

eval:
	python scripts/evaluate.py

infer:
	python src/inference/transcribe.py

test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/
