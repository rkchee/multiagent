FROM nvcr.io/nvidia/tensorflow:20.10-tf2-py3


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && \
      apt-get -y install sudo
RUN apt-get install -y --no-install-recommends apt-utils

RUN pip install --upgrade pip
RUN pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-1.1.0.dev0-cp36-cp36m-manylinux2014_x86_64.whl
RUN pip install setproctitle ray[debug] requests psutil gputil
RUN pip install pandas 

RUN sudo apt-get update
RUN sudo apt install -y --no-install-recommends ffmpeg
RUN pip install cryptography
RUN pip install google-api-python-client==1.7.8

RUN sudo apt-get install -y --no-install-recommends rsync
RUN sudo apt install -y --no-install-recommends screen

RUN sudo apt-get install -y --no-install-recommends ca-certificates

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends google-cloud-sdk

RUN sudo apt-get install -y --no-install-recommends kubectl

RUN sudo apt-get update
RUN sudo apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev
RUN pip install opencv-python

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install flask
RUN pip install numpy
RUN pip install requests
RUN pip install transformers
RUN pip install pygments
RUN pip install uvicorn
RUN pip install blist
RUN pip install scikit-learn
RUN pip install -U ray[serve] ray[tune] ray[rllib]
RUN pip install ray[rllib]
RUN pip install xlrd==1.2.0
# RUN git clone https://github.com/nth-opinion/nxtopinion.git

RUN pip install -U requests

RUN pip install pipenv
RUN pip install dask[complete]
COPY training_medical.xlsx /
COPY train_set.csv /
COPY test_set.csv /
COPY combinedxy.hdf5 /