.PHONY: build test lint docs clean

build:
	pip install -e .

test:
	pytest

lint:
	ruff check .
	mypy .

clean:
	rm -rf __pycache__ .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.o' -delete
	find . -type f -name '*.so' -delete

docs:
	@echo "Docs target coming soon"
