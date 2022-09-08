FROM conda/miniconda3-centos7
COPY environment.yml .
RUN conda env update --prune --file environment.yml
COPY setup.cfg .
COPY setup.py .
COPY coast .
RUN python -m pip install .
