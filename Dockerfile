FROM python:3.7-buster

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.6.0%2Bcpu-cp37-cp37m-linux_x86_64.whl

RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.7.0%2Bcpu-cp37-cp37m-linux_x86_64.whl

RUN pip install numpy==1.20.1 opencv-python==4.4.0.40 torchsummary==1.5.1 Pillow==8.1.0 matplotlib==3.3.4

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

ENV PYTHONPATH /repo

COPY recognizer ./recognizer

CMD [ "bash" ]
