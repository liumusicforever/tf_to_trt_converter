# base image
# FROM tensorflow/tensorflow:1.8.0-devel-gpu-py3

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update
RUN apt-get install -y python3.5
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get update


WORKDIR /root

ADD ./nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb /root

RUN dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
RUN apt-get update
RUN apt-get install -y tensorrt


RUN apt-get install -y python3-libnvinfer python3-libnvinfer-dev python3-libnvinfer-doc
RUN apt-get install -y uff-converter-tf
RUN pip3 install numpy
RUN pip3 install pycuda


RUN apt-get install -y  wget

RUN pip3 install tensorflow-gpu==1.8.0
RUN cp /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7

RUN pip3 install opencv-python
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev
RUN apt-get install -y libglib2.0



RUN pip3 install jupyter
RUN pip3 install argparse
RUN pip3 install enum34
RUN pip3 install Flask
RUN pip3 install future
RUN pip3 install PILLOW
RUN pip3 install future

