FROM jupyter/metadata_main

RUN pip install joblib

COPY data/fma_metadata/track.csv ./track.csv 

COPY src/metadata_pipe.py ./metadata_pipe.py
COPY src/audio_pipe ./audio_pipe

RUN python3 metadata_pipe.py
