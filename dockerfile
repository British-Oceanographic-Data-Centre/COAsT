FROM conda/miniconda3-centos7
RUN yum install wget unzip -y
RUN conda install python=3.8 cartopy
RUN wget https://linkedsystems.uk/erddap/files/COAsT_example_files/COAsT_example_files.zip
COPY coast ./coast
COPY setup.cfg .
COPY setup.py .
COPY notebook_to_md.sh .
RUN python -m pip install .
COPY example_scripts ./example_scripts
COPY config ./example_scripts/notebooks/config
RUN unzip COAsT_example_files.zip && mv COAsT_example_files ./example_scripts/notebooks/example_files
RUN bash notebook_to_md.sh
