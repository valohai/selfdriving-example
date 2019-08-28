FROM pytorch/pytorch
RUN pip install opencv-python numpy
RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
RUN apt-get update
RUN apt-get -y install libglib2.0-0 libsm6 libxext6 libxrender-dev