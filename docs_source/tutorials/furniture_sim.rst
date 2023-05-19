How to Use FurnitureSim
=======================

FurnitureSim Configuration
----------------------------

FurnitureSim can be configured with the following arguments:

.. code::

    import gym
    import furniture_bench

    env = gym.make(
      "Furniture-Sim-Env-v0",
      furniture=...,           # Specifies the type of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
      num_envs=1,              # Number of parallel environments.
      resize_img=True,         # If true, images are resized to 224 x 224.
      headless=False,          # If true, simulation runs without GUI.
      compute_device_id=0,     # GPU device ID for simulation.
      graphics_device_id=0,    # GPU device ID for rendering.
      init_assembled=False,    # If true, the environment is initialized with assembled furniture.
      np_step_out=False,       # If true, env.step() returns Numpy arrays.
      channel_first=False,     # If true, images are returned in channel first format.
      randomness='low',        # Level of randomness in the environment [low | med | high].
      high_random_idx=-1,      # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
      save_camera_input=False, # If true, the initial camera inputs are saved.
      record=False             # If true, videos of the wrist and front cameras' RGB inputs are recorded.
    )

Parameters
~~~~~~~~~~~~~~
- ``randomness`` controls the randomness level of the environment. This argument operates identically as FurnitureBench (See "Key Parameters" section in :ref:`How to use FurnitureBench`).

|
  We show how parameters affect the environment with example codes below:

- To check meshes and physics parameters of a furniture model, you can initialize FurnitureSim with the fully assembled furniture using ``--init-assembled``:

  .. code:: bash

    python furniture_bench/scripts/run_sim_env.py --furniture <furniture> --init-assembled
    # E.g.
    python furniture_bench/scripts/run_sim_env.py --furniture chair --init-assembled

.. figure:: ../_static/images/chair_assembled.jpg
    :width: 450px

    Initialization with an assembled chair.

- You can save camera input images at the beginning of the episode using ``--save-camera-input``. The output images will be saved in ``sim_camera/`` directory.

  .. code:: bash

     python furniture_bench/scripts/run_sim_env.py --furniture <furniture> --init-assembled --save-camera-input
     # E.g.
     python furniture_bench/scripts/run_sim_env.py --furniture square_table --init-assembled --save-camera-input


  .. |image1| image:: ../_static/images/wrist_sim.png
      :width: 215px
      :height: 120px
  .. |image2| image:: ../_static/images/front_sim.png
      :width: 215px
      :height: 120px
  .. |image3| image:: ../_static/images/rear_sim.png
      :width: 215px
      :height: 120px

  +--------------+--------------+-------------+
  | |image1|     | |image2|     |  |image3|   |
  +==============+==============+=============+
  | Wrist camera | Front camera | Rear camera |
  +--------------+--------------+-------------+

- The ``--record`` flag can be used to capture the wrist and front camera during the entire episode. The resulting videos will be saved in ``sim_record/`` directory.

  Here's an instance of an episode recording for an automated script:

  .. figure:: ../_static/images/wrist_and_front.gif

Automated Assembly Script
~~~~~~~~~~~~~~~~~~~~~~~~~

We provide automated furniture assembly scrips. It currently supports only ``one_leg``.

..  ============== =================
..    Furniture     Assembly script
..  ============== =================
..       lamp              ⏳
..   square_table          ⏳
..       desk              ⏳
..   round_table           ⏳
..      stool              ⏳
..      chair              ⏳
..      drawer             ⏳
..     cabinet             ⏳
..     one_leg             ✔️
..  ============== =================

.. code:: bash

   python furniture_bench/scripts/run_sim_env.py --furniture one_leg --scripted

.. figure:: ../_static/images/assembly_script.gif
    :width: 50%
    :align: left
    :alt: Assembly script

.. tip::

    On your initial run, starting up the simulator will take some time because it needs to construct SDF meshes.
    However, following runs will be much quicker as the simulator will load the cached SDF meshes.


Using this assembly script, you can collect demonstration data:

.. code:: bash

   python furniture_bench/scripts/collect_data.py --furniture <furniture> --scripted --is-sim --out-data-path <path/to/output> --gpu-id <gpu_id> --headless  # Make sure you mount the output data path to the docker container.
   # E.g.
   python furniture_bench/scripts/collect_data.py --furniture one_leg --scripted --is-sim --out-data-path /hdd/scripted_sim_demo  --gpu-id 0 --headless
   # You can specify the number of trajectories to collect in the run with --num-demos argument.
   # E.g.
   python furniture_bench/scripts/collect_data.py --furniture one_leg --scripted --is-sim --out-data-path /hdd/scripted_sim_demo  --gpu-id 0 --headless --num-demos 300

To visualize a collected demonstration, use the following script with a demonstration path:

.. code:: bash

   python furniture_bench/scripts/show_trajectory.py --data-dir <path/to/saved/data/dir>
   # E.g.
   python furniture_bench/scripts/show_trajectory.py --data-dir /hdd/scripted_sim_demo/one_leg/2022-12-22-03:19:48


Teleoperation in FurnitureSim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FurnitureSim supports teleoperation using a keyboard and Oculus Quest 2.
You first need to set up Oculus Quest 2 by following :ref:`Teleoperation`.

To start FurnitureSim with teleoperation, execute the following command:

.. code::

    python furniture_bench/scripts/collect_data.py --furniture <furniture> --out-data-path <path/to/save/data> --input-device oculus --is-sim
