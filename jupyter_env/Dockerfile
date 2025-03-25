FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG USERNAME=lusha
ARG USER_UID=1142
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND noninteractive
# add repo for opython 3.X
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa 

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    wget \
    htop \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
	python3 \
	python3-dev \
	python3-distutils \
	python3-pip \
	vim \
	cmake \
	glibc-source \
	sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -g $USER_GID $USERNAME \
    # [Optional] Remove the following line if you don't want to use the second group. Remove also the "--groups $GROUP2_NAME" option in the useradd command below.
	&& useradd -u $USER_UID -g $USER_GID  -m $USERNAME \
	#
	# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
	&& apt-get install -y sudo \
	&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME


USER $USERNAME
WORKDIR /home/$USERNAME

# -- Create folders under root for following services
USER 0
# Enable jupyter
RUN mkdir -p /.local
RUN chmod -R 777 /.local

USER $USERNAME
RUN mkdir -p /home/$USERNAME/.local/bin
RUN chmod -R 777 /home/$USERNAME/.local/bin
ADD requirements.txt /requirements.txt

ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PATH="/.local/bin:${PATH}"

USER 0
RUN chown $USERNAME /requirements.txt


USER $USERNAME 
RUN pip install -r /requirements.txt

# # Install PyTorch with CUDA 12.1 support
# RUN pip3 install --no-cache-dir \
#     torch==2.1.0+cu121 \
#     torchvision==0.16.0+cu121 \
#     torchaudio==2.1.0+cu121 \
#     --extra-index-url https://download.pytorch.org/whl/cu121

#RUN mkdir .jupyter
#RUN echo c.NotebookApp.token = '' > .jupyter/jupyter_notebook_config.py \
#	&& echo c.NotebookApp.password = '' >> .jupyter/jupyter_notebook_config.py



USER 0
# Install FFmpeg and basic dependencies
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libavcodec-dev \
		libavcodec-extra \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        && rm -rf /var/lib/apt/lists/*

RUN ls /usr/local/cuda-11.7/targets/x86_64-linux/lib/libnpp*

# Install NVIDIA Video Codec SDK headers
RUN mkdir -p /usr/local/include/ffmpeg && \
    cd /usr/local/include/ffmpeg && \
    wget -q https://github.com/FFmpeg/nv-codec-headers/releases/download/n11.1.5.1/nv-codec-headers-11.1.5.1.tar.gz && \
    tar xzf nv-codec-headers-11.1.5.1.tar.gz && \
    cd nv-codec-headers-11.1.5.1 && \
    make install && \
    rm -rf /usr/local/include/ffmpeg/nv-codec-headers-11.1.5.1*

USER $USERNAME
ENV SHELL /bin/bash

EXPOSE 8888

#RUN ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
CMD ["bash"]