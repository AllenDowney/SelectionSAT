PROJECT_NAME = SelectionSAT
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

create_environment:
	conda create -y -c conda-forge --name $(PROJECT_NAME) python pymc
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

delete_environment:
	conda env remove --name $(PROJECT_NAME)

requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 code
	black --check --config pyproject.toml code

format:
	black --config pyproject.toml code

tests:
	pytest --nbmake *.ipynb
