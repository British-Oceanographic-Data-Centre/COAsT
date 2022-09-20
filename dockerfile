FROM conda/miniconda3-centos7
RUN conda install python=3.8 cartopy
COPY setup.cfg .
COPY setup.py .
COPY coast ./coast
RUN python -m pip install .
COPY config ./config
RUN pip install jupyter

