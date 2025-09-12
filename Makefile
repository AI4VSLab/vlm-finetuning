PYTHON = uv run

.PHONY: install
install:
	uv sync --python python3.10 --all-groups

.PHONY: train
train:
	export MPLBACKEND=agg && $(PYTHON) src/train.py

.PHONY: test
test:
	export MPLBACKEND=agg && $(PYTHON) src/test.py

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delet
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
	$(PYTHON) -m ruff clean

.PHONY: lint
lint:
	$(PYTHON) -m ruff check . --fix

.PHONY: format
format:
	$(PYTHON) -m ruff format .
