FROM docker-repo.bodc.me/bodc/coast:base
ADD requirements_docs.txt requirements_docs.txt
ADD docstring2md-0.4.1-py3-none-any.whl docstring2md-0.4.1-py3-none-any.whl
RUN pip install -r requirements_docs.txt
