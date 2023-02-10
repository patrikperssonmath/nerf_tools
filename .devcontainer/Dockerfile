
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

#RUN pip install git+https://github.com/princeton-vl/lietorch.git

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    tzdata \
    dirmngr \
    gnupg2 \    
    wget \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6 \
    nano \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# install build packages

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    build-essential \
    libcurl4-gnutls-dev \
    libpng++-dev \
    cmake \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm -rf /var/lib/apt/lists/*

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

VOLUME /database
VOLUME /workspace
VOLUME /models
VOLUME /results
VOLUME /database_ssd
VOLUME /splits

ENV DEBIAN_FRONTEND=dialog

ENV GRPC_POLL_STRATEGY epoll1

RUN pip uninstall pillow -y
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

RUN PYBIND11_GLOBAL_SDIST=1 python3 -m pip install https://github.com/pybind/pybind11/archive/master.zip

RUN python3 -m pip install sophuspy

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN echo "PATH='$PATH:/home/$USERNAME/.local/bin'" >> /home/$USERNAME/.bashrc

RUN pip install git+https://github.com/patrikperssonmath/geometry3d.git
RUN pip install git+https://github.com/patrikperssonmath/SFMVisualization.git