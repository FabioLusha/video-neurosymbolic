FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG USERNAME=lusha
ARG USER_UID=1142
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND noninteractive
# add repo for opython 3.X
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa 

RUN apt-get update && apt-get install -y \
	build-essential ca-certificates python3 python3-dev python3-distutils git vim wget cmake python3-pip  glibc-source 

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


ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# -- Create folders under root for following services
USER 0
# Enable jupyter
RUN mkdir -p /.local
RUN chmod -R 777 /.local

USER $USERNAME

ADD requirements.txt /requirements.txt

USER 0
RUN chown $USERNAME /requirements.txt
USER $USERNAME 

USER 0
RUN apt install -y libsm6 libxext6 libxrender-dev ffmpeg htop

USER $USERNAME 
RUN pip install -r /requirements.txt

#RUN mkdir .jupyter
#RUN echo c.NotebookApp.token = '' > .jupyter/jupyter_notebook_config.py \
#	&& echo c.NotebookApp.password = '' >> .jupyter/jupyter_notebook_config.py

ENV SHELL /bin/bash

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
