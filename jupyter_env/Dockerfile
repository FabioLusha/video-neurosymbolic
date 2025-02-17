FROM python:3.8.10

ARG USERNAME=lusha
ARG USER_UID=$UID
ARG USER_GID=$GID

WORKDIR /home/$USERNAME

# COPY $src $dest, wher dest is a dir or file inside the container
# and the path of dest is relative to WORKDIR
COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt

ENV SHELL /bin/bash
EXPOSE 8888

CMD ["bash"]