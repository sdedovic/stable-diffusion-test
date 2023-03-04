FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip3 install keras-cv tensorflow_datasets

# because I want to load the hugginface textual-inversion
#  embeddings, and they pickle, zip, and depend on torch utils
RUN pip3 install torch 
