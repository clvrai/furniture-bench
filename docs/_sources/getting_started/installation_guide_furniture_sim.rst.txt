Installation Guide (FurnitureSim)
=================================

**FurnitureSim** is a realistic simulation of FurnitureBench based on `Isaac Gym <https://developer.nvidia.com/isaac-gym>`__ and `Factory <https://sites.google.com/nvidia.com/factory>`__.
FurnitureSim enables easy and fast evaluation of new algorithms.
You can install FurnitureSim using Docker or build it from source.

.. |real| image:: ../_static/images/real.jpg
.. |simulator| image:: ../_static/images/simulator.jpg
.. |rendering| image:: ../_static/images/rendering.jpg

.. table::
    :widths: 30 30 30

    +------------------+------------------------+-----------------------------+
    | |simulator|      |    |rendering|         |          |real|             |
    +==================+========================+=============================+
    |  \(a) Simluator  | \(b) Offline rendering | \(c) Real-world environment |
    +------------------+------------------------+-----------------------------+


.. prerequisites::
    Prerequisites

    -  🛠️ Ubuntu 20.04 LTS
    - NVIDIA RTX GPU
    -  📖 `Anaconda <https://www.anaconda.com/>`__


Download Isaac Gym
~~~~~~~~~~~~~~~~~~

1. Go to the `Isaac Gym website <https://developer.nvidia.com/isaac-gym>`__.
2. Click "Join now" and log into your NVIDIA account.
3. Click "Member area".
4. Read and check the box for the license agreement.
5. Download and unzip ``Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release``.


Install FurnitureSim using Docker (Option 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our client Docker image includes FurnitureSim:

1. Install `nvidia-docker2 <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`__.

2. Specify whether to pull the Docker image from Docker Hub or build it locally:

.. code:: bash

  # Case 1: pull from Docker Hub
  export CLIENT_DOCKER=furniturebench/client-gpu:latest

  # Case 2: build locally
  export CLIENT_DOCKER=client-gpu

3. Set the environment variables following :ref:`Run Client`.

.. code:: bash

  # With display
  xhost +

  # Set the absolute path to the furniture-bench repo
  export FURNITURE_BENCH=<path/to/furniture-bench>

  # Set the absolute path to Isaac Gym
  export ISAAC_GYM_PATH=<path/to/isaacgym>

4. Launch the Docker image:

.. code::

  ./launch_client.sh --sim-gpu


Install FurnitureSim from Source (Option 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can manually install FurnitureSim and its dependencies as follows:

1. Install CUDA following the instructions `here <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__ or this `gist <https://gist.github.com/primus852/b6bac167509e6f352efb8a462dcf1854#file-cuda_11-7_installation_on_ubuntu_22-04>`__. You can skip this step if you already have CUDA installed on your machine.

2. Create a conda environment:

.. code::

  conda create -n furniture-bench python=3.8

3. Install the PyTorch version you want to use:

.. code::

  conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c conda-forge

4. Install Isaac Gym:

.. code::

  cd <path/to/isaacgym>
  cd python
  pip install -e .

5. Install FurnitureBench, which includes FurnitureSim:

.. code::

  git clone https://github.com/clvrai/furniture-bench.git
  cd furniture-bench
  pip install -e .

  # Match the version of setuptools with the packages in the requirements.txt
  pip install --upgrade pip wheel
  pip install setuptools==58
  pip install --upgrade pip==22.2.2

  pip install -r requirements.txt


Test FurnitureSim Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Test FurnitureSim using the following command with a furniture name ``<furniture>=[lamp|square_table|desk|drawer|cabinet|round_table|stool|chair|one_leg]``.

.. code:: bash

  python furniture_bench/scripts/run_sim_env.py --furniture <furniture> --no-action

- You can also run our scripted agent for ``one_leg`` by adding ``--scripted`` to the arguments:

.. code:: bash

  python furniture_bench/scripts/run_sim_env.py --furniture one_leg --scripted


.. note::

    FurnitureSim needs to convert 3D furniture meshes into Signed Distance Functions (SDF), which takes several minutes. The converted SDF models are cached for fast execution.


FurnitureSim Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The simulation parameters (e.g., mass, inertia, and dt) can be found in ``furniture_bench/sim_config.py``.
