FROM docker-repo.bodc.me/bodc/conda:latest

RUN /home/bodc/miniconda3/bin/conda install -c conda-forge gcc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
