PROJECT_NAME = SelectionSAT

create_environment:
	mamba env create -f environment.yml
	@echo ">>> Environment created. Activate with: mamba activate $(PROJECT_NAME)"

update_environment:
	mamba env update -f environment.yml --prune

delete_environment:
	mamba env remove --name $(PROJECT_NAME)

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# Regenerate the jupytext copies in md/ after changing a notebook
md:
	python convert_notebooks_to_md.py

zip:
	zip sat_math_csv_files.zip sat_math_20[0-9][0-9].csv

.PHONY: create_environment update_environment delete_environment clean md zip
