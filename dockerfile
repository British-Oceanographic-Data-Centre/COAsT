FROM conda/miniconda3-centos7
RUN conda install python=3.8
COPY setup.cfg .
COPY setup.py .
COPY coast ./coast
RUN python -m pip install .
