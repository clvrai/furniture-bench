FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_NAME=client
ARG LIBFRANKA_VERSION=0.9.2
ARG PYTHON_VERSION=3.8

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

# Install common.
RUN apt-get update \
    && apt-get install -y --no-install-recommends vim bzip2 wget ssh unzip git iproute2 iputils-ping build-essential curl \
    cmake ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libeigen3-dev libopenblas-dev libpoco-dev libeigen3-dev libspdlog-dev

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

# Install pyrealsense2 and dt-apriltags.
RUN apt update \
    && apt install -y libcanberra-gtk-module libcanberra-gtk3-module libusb-1.0-0-dev \
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
    && apt-get install -y kmod kbd \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install keyboard

# Setup Oculus
RUN apt update && apt install -y android-tools-adb \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install git-lfs \
    && git lfs install \
    && /opt/conda/envs/${VENV_NAME}/bin/pip install git+https://github.com/rail-berkeley/oculus_reader.git

# Install PyTorch
RUN /opt/conda/condabin/conda install -n ${VENV_NAME} pytorch=1.10 cpuonly -c conda-forge -c pytorch

# Install EigenPy
RUN apt-get update \
    && apt-get install -y lsb-release && apt-get clean all

RUN sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list" \
    && curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | apt-key add - \
    && apt-get update \
    && apt install -y robotpkg-py38-eigenpy

RUN git clone --recursive https://github.com/stack-of-tasks/pinocchio --branch v2.6.17 \
    && apt-get update \
    && export CMAKE_PREFIX_PATH=/opt/openrobots \
    && apt-get install -y --no-install-recommends liburdfdom-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libboost-filesystem-dev \
    && cd pinocchio && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DPYTHON_EXECUTABLE=/opt/conda/envs/${VENV_NAME}/bin/python \
    && make -j4 \
    && make install

SHELL ["/opt/conda/bin/conda", "run", "-n", "client", "/bin/bash", "-c"]
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

RUN /opt/conda/envs/${VENV_NAME}/bin/pip install -U pip setuptools
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install setuptools==58
# Install numpy 1.23.5 to fix the issue of "from numba.np.ufunc import _internal SystemError: initialization of _internal failed without raising an exception"
RUN /opt/conda/envs/${VENV_NAME}/bin/pip install numpy==1.23.5

# Copy the entrypoint.sh script to the Docker image
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint.sh script executable
RUN chmod +x /entrypoint.sh

# Run the entrypoint.sh script and start the bash shell
CMD ["/entrypoint.sh"]
