Installation Guide (FurnitureSim)
=================================

FurnitureSim is a high-speed and realistic simulation of FurnitureBench based on Isaac Gym and Factory.
It is designed to be a seamless substitution of the real-world environment, which enables easy and fast evaluation of new algorithms.
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
    -  📖 Download `Isaac Gym <https://developer.nvidia.com/isaac-gym>`__
    -  📖 `Anaconda <https://www.anaconda.com/>`__


Download Isaac Gym
~~~~~~~~~~~~~~~~~~

FurnitureSim is built on top of Isaac Gym and Factory; thus, you need to download Isaac Gym:

1. Go to the `website <https://developer.nvidia.com/isaac-gym>`__.
2. Start by creating an NVIDIA account, and then click "Member area".
3. Kindly review and accept the "Terms of the NVIDIA Isaac Gym License Agreement".
4. Download ``Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release``.
5. Unzip the downloaded file and move the folder to the desired location.


Install FurnitureSim using Docker (Option 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a docker image that includes everything needed for FurnitureSim. To use FurnitureSim, you only need to run our docker image.

1. 📖 Download `nvidia-docker2 <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`__.

2. Set the environment variables by following :ref:`Run Client`.

  .. code:: bash

    # With display
    xhost +

    export CLIENT_DOCKER=client-gpu                        # (Case1: local build)
    export CLIENT_DOCKER=furniturebench/client-gpu:latest  # (Case2: docker pull)

    # Set the absolute path to the furniture-bench repo
    export FURNITURE_BENCH=</path/to/furniture-bench>

    # Set the absolute path to Isaac Gym
    export ISAAC_GYM_PATH=</path/to/isaacgym>

3. Launch the Docker image:

  .. code::

    ./launch_client --sim-gpu

4. Test FurnitureSim:

  .. code:: bash

    # --furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg]

    python furniture_bench/scripts/run_sim_env.py --furniture square_table --no-action [--headless]



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

    cd </path/to/isaacgym>
    cd python
    pip install -e .

5. Install FurnitureBench:

  .. code::

    git clone https://github.com/clvrai/furniture-bench.git
    cd furniture-bench
    pip install -e .

    # Match the version of setuptools with the packages in the requirements.txt
    pip install --upgrade pip wheel
    pip install setuptools==58
    pip install --upgrade pip==22.2.2

    pip install -r requirements.txt

6. Test FurnitureSim:

  .. code::

    python furniture_bench/scripts/run_sim_env.py --furniture one_leg --scripted


.. note::

    FurnitureSim needs to convert 3D furniture meshes into Signed Distance Functions (SDF), which takes several minutes. The converted SDF models are cached for fast execution.


FurnitureSim Parameters
~~~~~~~~~~~~~~~~~~~~~~~
Simulation parameters (e.g., mass, inertia, and dt) can be found in ``furniture_bench/sim_config.py``.


Furniture Assembly Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~

We are planning to provide hard-coded furniture assembly scrips for expert data collection. Currently, FurnitureSim includes a script only for ``one_leg``.

 ============== =================
   Furniture     Assembly script
 ============== =================
      lamp              ⏳
  square_table          ⏳
      desk              ⏳
  round_table           ⏳
     stool              ⏳
     chair              ⏳
     drawer             ⏳
    cabinet             ⏳
    one_leg             ✔️
 ============== =================
