FROM jupyter/scipy-notebook

USER root
# Add permanent apt-get installs and other root commands here

USER jovyan
# Add permanent pip/conda installs, data files, other user libs here

RUN conda install --quiet --yes -c \
    https://conda.anaconda.org/jjhelmus tensorflow \
    && conda clean -tipsy

RUN conda install --quiet --yes \
    'psycopg2=2.6*' \
    'sqlalchemy=1.0*' \
    && conda clean -tipsy


# Prepare font cache
RUN python -c "import matplotlib.pyplot"