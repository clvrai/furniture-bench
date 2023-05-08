FurnitureSim
===================

Assembly Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we show how to run automated assembly scripts that is based on the finite state machine (FSM).

.. code:: bash

   python furniture_bench/scripts/run_sim_env.py --furniture one_leg --scripted # Note: only one_leg is supported for now.

.. tip::

    On your initial run, starting up the simulator will take some time because it needs to construct SDF meshes.
    However, following runs will be much quicker as the simulator will load the cached SDF meshes.

You will see something like this:

.. figure:: ../../_static/images/assembly_script.gif
    :width: 50%
    :align: left
    :alt: Assembly script


Environment with Assembled Furniture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we show how to initialize the simulation environment with already assembled furniture.
This is useful to check whether the simulator can correctly simulate contacts of the assembled furniture.

.. code:: bash

   python furniture_bench/scripts/run_sim_env.py --furniture drawer --init-assembled

   # args: --furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg]

Data Collection Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this section, we demonstrate the process of data collection using the automated assembly scripts discussed earlier.

.. code:: bash

   python furniture_bench/scripts/collect_data.py --furniture {furniture} --scripted --is-sim --out-data-path {path/to/output} --gpu-id {gpu_id} --headless # Make sure you mount the output data path to the docker container.

   # e.g.,
   python furniture_bench/scripts/collect_data.py --furniture one_leg --scripted --is-sim --out-data-path /hdd/IL_data_sim --gpu-id 0 --headless

   # To visualize saved data.
   python furniture_bench/scripts/show_trajectory.py --data-dir {path/to/saved/data/dir}
   # e.g.,
   python furniture_bench/scripts/show_trajectory.py --data-dir /hdd/IL_data_sim/one_leg/2022-12-22-03:19:48

Teleoperation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulation also supports teleoperation using keyboard and Oculus Quest 2.
The basic setup and commands are the same as the real robot :ref:`Teleoperation using Oculus/Keyboard`

.. prerequisites::
    Prerequisites

    - :ref:`Setup Oculus Quest 2`

Run the following command to start the simulation environment with teleoperation.

.. code::

    python furniture_bench/scripts/collect_data.py --furniture {furniture} --out-data-path {path/to/save/data} --input-device oculus --is-sim


Save Initial Camera Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This will save the camera input images in the beginning of the episode, which is useful for debugging the visual input.
The front, wrist, and rear images will be saved to the ``sim_camera/`` directory.

.. code:: bash

   python furniture_bench/scripts/run_sim_env.py --furniture square_table --init-assembled --save-camera-input


.. |image1| image:: ../../_static/images/wrist_sim.png
    :width: 215px
    :height: 120px
.. |image2| image:: ../../_static/images/front_sim.png
    :width: 215px
    :height: 120px
.. |image3| image:: ../../_static/images/rear_sim.png
    :width: 215px
    :height: 120px

+--------------+--------------+-------------+
| Wrist camera | Front camera | Rear camera |
+==============+==============+=============+
| |image1|     | |image2|     |  |image3|   |
+--------------+--------------+-------------+

.. Policy Training and Evaluation
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. code::

..     # BC conversion
..     python furniture_bench/scripts/convert_data.py --in-data-path /data/minho/hdd/IL_data/one_leg/ --out-data-path /data/minho/converted_one_leg_mixed_2000/

..     # IQL conversion
..     python implicit_q_learning/convert_furniture_data.py --furniture one_leg --demo_dir /hdd/converted_stool_full_100 --out_file_path one_leg_sim.pkl --use_r3m
.. TODO

Additional Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. seealso::

    Isaac Gym provides a documentation along with the source code.
    Thus, run ``xdg-open $ISAAC_GYM_PATH/docs/index.html``.
    It will open the documentation on your browser that contains more information about concepts, API, and examples.
