FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV VENV_NAME=client-gpu
ARG LIBFRANKA_VERSION=0.9.2
ARG PYTHON_VERSION=3.8

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

# Install common.
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim bzip2 wget ssh unzip git iproute2 iputils-ping build-essential curl \
    cmake ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libpoco-dev libeigen3-dev libspdlog-dev libopenblas-dev \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev mesa-common-dev make gcc-8 g++-8 vulkan-utils mesa-vulkan-drivers pigz libegl1

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install Polymetis.
RUN git clone --recursive https://github.com/minoring/fairo.git --branch furniture \
    && cd /fairo/polymetis \
    && /opt/conda/bin/conda env create -f ./polymetis/environment.yml -n ${VENV_NAME} python=${PYTHON_VERSION} \
    && echo "conda activate ${VENV_NAME}" >> ~/.bashrc

# Installation of robot learning framework
RUN apt-get update \
    # Install mujoco
    && mkdir /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O /root/.mujoco/mujoco210_linux.tar.gz \
    && tar -xvzf /root/.mujoco/mujoco210_linux.tar.gz -C /root/.mujoco/ \
    && rm /root/.mujoco/mujoco210_linux.tar.gz \
    # download MuJoCo 2.1.1 for dm_control
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O /root/.mujoco/mujoco211_linux.tar.gz \
    && tar -xvzf /root/.mujoco/mujoco211_linux.tar.gz -C /root/.mujoco/ \
    && rm /root/.mujoco/mujoco211_linux.tar.gz \
    # add MuJoCo 2.1.0 to LD_LIBRARY_PATH
    && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin" >> ~/.bashrc \
    # for GPU rendering
    && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc \
    && apt-get install -y --no-install-recommends cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev \
    # software rendering
    && apt-get install -y --no-install-recommends libgl1-mesa-glx libosmesa6 patchelf \
    # window rendering
    && apt-get install -y --no-install-recommends libglfw3 libglew-dev \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin \
    && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install install mujoco_py \
    # trigger mujoco to compile
	&& /opt/conda/envs/${VENV_NAME}/bin/python -c "import mujoco_py"

# Install pyrealsense2 and dt-apriltags.
RUN apt update \
    && apt install -y --no-install-recommends libcanberra-gtk-module libcanberra-gtk3-module libusb-1.0-0-dev \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install pyrealsense2
RUN mkdir /wheels
COPY wheels/dt_apriltags-3.2.0-py3-none-manylinux2010_x86_64.whl /wheels
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install /wheels/dt_apriltags-3.2.0-py3-none-manylinux2010_x86_64.whl

# Install python packages.
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install pip==22.2.2
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install setuptools==65.5.0
RUN /opt/conda/condabin/conda install -n ${VENV_NAME} numpy
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install joblib h5py==3.6.0 opencv-python==4.1.2.30 numba
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install gym==0.21.0

# Install keyboard
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends kmod kbd \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install keyboard

# Setup Oculus
RUN apt update && apt install -y --no-install-recommends android-tools-adb \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install --no-install-recommends git-lfs \
    && git lfs install \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install git+https://github.com/rail-berkeley/oculus_reader.git

RUN /opt/conda/envs/${VENV_NAME}/bin/pip install --upgrade hydra-core
# RUN /opt/conda/condabin/conda install -n ${VENV_NAME} pytorch==1.12 torchvision cudatoolkit=11.7 -c pytorch
# RUN /opt/conda/condabin/conda install -n ${VENV_NAME} pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN /opt/conda/condabin/conda install -n ${VENV_NAME} pytorch==1.10 torchvision cudatoolkit=11.3 -c pytorch

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    # Add robopkg
    && apt install -qqy lsb-release gnupg2 \
    && echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" | tee /etc/apt/sources.list.d/robotpkg.list \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - \
    && apt-get update \
    && apt install -qqy robotpkg-py38-pinocchio

# Build Polymetis
SHELL ["/opt/conda/bin/conda", "run", "-n", "client-gpu", "/bin/bash", "-c"]
RUN cd fairo/polymetis \
    # Add pinnochio path.
    && export PATH=/opt/openrobots/bin:$PATH \
    && export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH \
    && export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH \
    && export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH \
    && export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install -e ./polymetis \
    && cd /fairo \
    && git submodule update --init --recursive \
    && cd /fairo/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka \
    && git checkout ${LIBFRANKA_VERSION} \
    && git submodule update \
    && mkdir ./build \
    && cd ./build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF .. \
    && cmake --build . \
    && cd /fairo \
    && mkdir -p ./polymetis/polymetis/build \
    && cd ./polymetis/polymetis/build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF \
    && make -j
SHELL ["/bin/bash", "-c"]

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install IsaacGymEnvs
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install -U pip setuptools
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install setuptools==58
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install -e IsaacGymEnvs

# Reinstall Gym
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install pip==22.2.2
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install setuptools==65.5.0
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install gym==0.21.0

# Copy the entrypoint.sh script to the Docker image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint.sh script executable
RUN chmod +x /entrypoint.sh


# Disable memory allocation from JAX; otherwise it will cause out-of-memory (OOM) errors.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the entrypoint.sh script and start the bash shell
CMD ["/entrypoint.sh"]
