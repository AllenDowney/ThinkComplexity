PROJECT_NAME = ThinkComplexity
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

delete_environment:
	conda env remove --name $(PROJECT_NAME)

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 code
	black --check --config pyproject.toml code

format:
	black --config pyproject.toml code

# soln/ holds the canonical copies of the modules; nb/ holds copies, which are
# what the notebooks download from GitHub, so they have to be real files.
SHARED = $(notdir $(wildcard soln/*.py))

sync-shared:
	for f in $(SHARED); do cp soln/$$f nb/$$f; done

# Fails if a copy in nb/ is out of date.
check-shared:
	@for f in $(SHARED); do \
		cmp -s soln/$$f nb/$$f || { echo "nb/$$f is out of date: run make sync-shared"; exit 1; }; \
	done

# Student notebooks are not tested: their exercise cells are blank, so they
# stop at the first cell that calls a function the reader is meant to write.
# One pytest run; nbmake runs each notebook in its own directory.
tests: check-shared
	pytest --nbmake --durations=10 soln/*.ipynb
