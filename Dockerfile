FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip3 install keras-cv tensorflow_datasets
