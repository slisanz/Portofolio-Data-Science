.PHONY: install lint test app clean

install:
	pip install -r requirements.txt

lint:
	ruff check src tests app

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/interim/* data/processed/* models/*.joblib reports/figures/*.png
