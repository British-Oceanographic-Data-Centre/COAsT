FROM conda/miniconda3-centos7
RUN conda install python=3.8 cartopy
RUN pip install jupyterlab
COPY coast ./coast
COPY setup.cfg .
COPY setup.py .
RUN python -m pip install .
COPY example_files ./example_files
COPY example_scripts ./example_scripts
COPY config ./config