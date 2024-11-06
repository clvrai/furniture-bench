FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG LIBFRANKA_VERSION=0.13.3
ARG PYTHON_VERSION=3.8
ARG VENV_NAME=server
ARG SSH_PRIVATE_KEY

# Use bash as a default shell.
SHELL ["/bin/bash", "-c"]

# Install misc.
RUN apt-get update \
    && apt-get install -y vim bzip2 wget ssh unzip git iproute2 iputils-ping build-essential curl \
    cmake ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 tmux libpoco-dev libeigen3-dev libspdlog-dev

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

# Install Polymetis.
# RUN git clone --recursive https://github.com/facebookresearch/fairo.git
# RUN git clone --recursive https://github.com/minoring/fairo.git --branch furniture
COPY fairo-FR3 /fairo

RUN /opt/conda/bin/conda update -n base -c defaults conda \
    && /opt/conda/bin/conda install -n base -c conda-forge mamba \
    && export MAMBA_ROOT_PREFIX=/opt/conda \
    && cd /fairo/polymetis \
    && /opt/conda/condabin/conda config --add channels conda-forge \
    # && /opt/conda/bin/conda config --set channel_priority flexible \
    && /opt/conda/bin/mamba env create -f ./polymetis/environment.yml -n ${VENV_NAME} python=${PYTHON_VERSION} \
    && echo "conda activate ${VENV_NAME}" >> ~/.bashrc

RUN /opt/conda/envs/${VENV_NAME}/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc -o /etc/apt/keyrings/robotpkg.asc \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list \
    && apt update \
    && apt install -qqy robotpkg-py3*-pinocchio

SHELL ["/opt/conda/bin/conda", "run", "-n", "server", "/bin/bash", "-c"]

RUN apt-get update && apt install -y robotpkg-py38-hpp-fcl

# editable install polymetis
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

SHELL ["/bin/bash", "-c"]
