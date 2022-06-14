FROM docker-repo.bodc.me/bodc/conda:latest
RUN conda install -c conda-forge gcc cartopy
COPY --chown=bodc:bodc . .
COPY requirements.txt .
RUN python -m pip install --upgrade pip virtualenv
RUN python -m pip install -r requirements.txt
RUN pip install dask[complete]
RUN sh setup_environment.sh
RUN sh build.sh