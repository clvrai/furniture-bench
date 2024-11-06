FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
ENV VENV_NAME=client-gpu
ARG LIBFRANKA_VERSION=0.13.3
ARG PYTHON_VERSION=3.8
ARG SSH_PRIVATE_KEY

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

# Install common.
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim bzip2 wget ssh unzip git iproute2 iputils-ping build-essential curl \
    cmake ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libpoco-dev libeigen3-dev libspdlog-dev libopenblas-dev \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev mesa-common-dev make gcc-8 g++-8 vulkan-utils mesa-vulkan-drivers pigz libegl1


RUN apt-get update && apt-get install -y gcc-9 g++-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 50

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN mkdir -p /root/.ssh && \
    echo "$SSH_PRIVATE_KEY" | tr -d '\r' > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# RUN git clone --recursive https://github.com/facebookresearch/fairo.git
# RUN git clone --recursive https://github.com/minoring/fairo.git --branch furniture
COPY fairo-FR3 /fairo

# Install Polymetis.
RUN /opt/conda/bin/conda update -n base -c defaults conda \
    && /opt/conda/bin/conda install -n base -c conda-forge mamba \
    && export MAMBA_ROOT_PREFIX=/opt/conda \
    && cd /fairo/polymetis \
    && /opt/conda/condabin/conda config --add channels conda-forge \
    # && /opt/conda/bin/conda config --set channel_priority flexible \
    && /opt/conda/bin/mamba env create -f ./polymetis/environment.yml -n ${VENV_NAME} python=${PYTHON_VERSION} \
    && echo "conda activate ${VENV_NAME}" >> ~/.bashrc


RUN /opt/conda/envs/${VENV_NAME}/bin/pip install "cython<3"

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
    && /opt/conda/envs/${VENV_NAME}/bin/pip install mujoco_py \
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
    && GIT_LFS_SKIP_SMUDGE=1 /opt/conda/envs/${VENV_NAME}/bin/pip install git+https://github.com/rail-berkeley/oculus_reader.git

RUN /opt/conda/envs/${VENV_NAME}/bin/pip install --upgrade hydra-core
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y lsb-release
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc -o /etc/apt/keyrings/robotpkg.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list \
    && apt update \
    && apt install -qqy robotpkg-py3*-pinocchio

# Build Polymetis
SHELL ["/opt/conda/bin/conda", "run", "-n", "client-gpu", "/bin/bash", "-c"]

RUN apt-get update && apt install -y robotpkg-py38-hpp-fcl

RUN cd fairo/polymetis \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install -e ./polymetis \
    && export PATH=/opt/openrobots/bin:$PATH \
    && export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH \
    && export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH \
    && export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH \
    && export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH \
    && cd /fairo \
    && git submodule update --init --recursive \
    && cd /fairo/polymetis/polymetis/src/clients/franka_panda_client/third_party/libfranka \
    && git checkout ${LIBFRANKA_VERSION} \
    && cd /fairo/polymetis \
    && ./scripts/build_libfranka.sh ${LIBFRANKA_VERSION} && \
    mkdir -p ./polymetis/build && \
    cd ./polymetis/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=ON -DBUILD_TESTS=ON -DBUILD_DOCS=ON -DCMAKE_CXX_STANDARD=17 && \
    make -j

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

RUN /opt/conda/envs/${VENV_NAME}/bin/pip install ur_rtde atomics threadpoolctl av zarr robosuite
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install --upgrade PyOpenGL PyOpenGL_accelerate

# Copy the entrypoint.sh script to the Docker image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint.sh script executable
RUN chmod +x /entrypoint.sh


# Disable memory allocation from JAX; otherwise it will cause out-of-memory (OOM) errors.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run the entrypoint.sh script and start the bash shell
CMD ["/entrypoint.sh"]
