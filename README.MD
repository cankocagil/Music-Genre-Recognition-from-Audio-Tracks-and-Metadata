# Music Genre Recognition from Audio Tracks and Metadata



![alt text](https://github.com/cankocagil/Music-Genre-Recognition-from-Audio-Tracks-and-Metadata/blob/main/results/visual/figures/danceability.png?raw=true)

Automatic recognition of audio tracks from music waves has been a challenging task in Music Information Retrieval (MIR). The mathematical representation of natural music, composed of rhythm, tones, intervals, patterns, and harmonies, is a complex subject for human specialists as there is a geometric interpretation in the humming of the strings and inherent subjective nature. The discrimination of genres from purely audio tracks is stochastic by nature and deeply controversial as the genres share common statistical knowledge. For instance, the musical composition emerges from the intersection of multi-cultural genres, known as fusion genres, e.g., country rock is a fusion of country music and rock music. Contextually, the difficulty level of autonomous discrimination of genres is changing over time. Music trends are changing, making the music genre classification task temporally dynamic and introducing a higher degree of conceptual complexity. For the scope of the project, we designed a computational system of end-to-end learning mechanisms based on classical machine, ensemble and deep learning algorithms for automatic recognition of music genres from Creative Commons licensed audio files, with hand-crafted track-level features.

## Our main study 
    * Metadata-level descriptive data analysis is applied.
    * Then, manifold learning and dimensionality reduction methods are performed on the metadata-level music data.
    * We built an end-to-end discovery machine learning pipelines to classify the music genres based on the metadata.
    * Audio-level discovery signal analysis methods are applied.
    * To convert audios to images, Mel Spectrum computation is performed.
    * 2-D Convolutional Neural Networks and Vision Transformer are employed to model music genres by digesting the state-of-the-art.
    

# Installation

We provide installation quidelines for this repo.

First, clone our project page as follows.

```
git clone https://github.com/cankocagil/Music-Genre-Recognition-from-Audio-Tracks-and-Metadata.git
```
We highly recommend the installation of PyTorch and Torchvision beforehand:

```
conda install -c pytorch pytorch torchvision
```

Then, install packages from requirements file:

```
pip install -r requirements.txt
```


# Data Preparation
Download and extract small version of FMA dataset from [official github page](https://github.com/mdeff/fma). 

```
Music Genre Recognition from Audio Tracks and Metadata
├── data
│   ├── fma_metadata
│   ├── fma_small
├── notebooks
...
```

We highly recommend the installation of PyTorch and Torchvision beforehand:

# Run

```
python metadata_pipe.py && python audio_pipe.py
```
