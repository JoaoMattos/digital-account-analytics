.PHONY: install run dashboard test lint
install:
	python -m pip install -r requirements.txt
run:
	python -m src.pipeline
dashboard:
	python -m streamlit run app.py
test:
	python -m pytest -q
lint:
	python -m ruff check .
