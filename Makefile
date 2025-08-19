PYTHON = uv run

.PHONY: install
install:
	uv sync --all-groups

.PHONY: pretrain
pretrain:
	export MPLBACKEND=agg && $(PYTHON) src/pretrain.py

.PHONY: finetune
finetune:
	export MPLBACKEND=agg && $(PYTHON) src/finetune.py

.PHONY: evaluate
evaluate:
	export MPLBACKEND=agg && $(PYTHON) src/evaluation.py

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
