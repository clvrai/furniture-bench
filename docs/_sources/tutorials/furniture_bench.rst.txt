How to Use FurnitureBench
=========================


FurnitureBench Environments
---------------------------

The following environments are available in FurnitureBench:
  * ``FurnitureBench-v0``: is mainly used for data collection, providing all available observations, including robot states, high-resolution RGB images, and depth inputs from wrist, front, and rear cameras.
  * ``FurnitureBenchImage-v0``: is used for pixel-based RL and IL by providing 224x224 wrist and front RGB images and robot states for observation.
  * ``FurnitureBenchImageFeature-v0``: provides pre-trained image features (R3M or VIP) instead of visual observations.
  * ``FurnitureDummy-v0``: Dummy environment for pixel-based policies.
  * ``FurnitureImageFeatureDummy-v0``: Dummy environment for policies with pre-trained visual encoders.


FurnitureBench Configuration
----------------------------

FurnitureBench can be configured with the following arguments:

.. code::

    import furniture_bench
    import gym

    env = gym.make(
      "FurnitureBench-v0",
      furniture=...,            # Specifies the name of furniture [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
      resize_img=True,          # If true, images are resized to 224 x 224.
      manual_done=False,        # If true, the episode ends only when the user presses the 'done' button.
      with_display=True,        # If true, camera inputs are rendered on environment steps.
      draw_marker=False,        # If true and with_display is also true, the AprilTag marker is rendered on display.
      manual_label=False,       # If true, manual labeling of the reward is allowed.
      from_skill=0,             # Skill index to start from (range: [0-5)). Index `i` denotes the completion of ith skill and commencement of the (i + 1)th skill.
      to_skill=-1,              # Skill index to end at (range: [1-5]). Should be larger than `from_skill`. Default -1 expects the full task from `from_skill` onwards.
      randomness="low",         # Level of randomness in the environment [low | med | high].
      high_random_idx=-1,       # Index of the high randomness level (range: [0-2]). Default -1 will randomly select the index within the range.
      visualize_init_pose=True, # If true, the initial pose of furniture parts is visualized.
      record=False,             # If true, the video of the agent's observation is recorded.
      manual_reset=True         # If true, a manual reset of the environment is allowed.
    )


FurnitureBench Arguments
~~~~~~~~~~~~~~~~~~~~~~~~

- ``furniture`` can be one of ``[lamp|square_table|desk|drawer|cabinet|round_table|stool|chair|one_leg]``.

- ``randomness`` controls the randomness level of the initial furniture and robot configurations.

  - For the ``med`` and ``high``, the end-effector pose is perturbed from the pre-defined target pose with noise (±5 cm positional, ±15◦ rotational).
  - For the ``low`` of the full assembly task, the end-effector pose is fixed to the pre-defined target pose.
  - For the ``low`` of the skill benchmark, the noise is applied to the pre-defined target pose (±0.5 cm positional, ±5◦ rotational).

.. figure:: ../_static/images/initialization_example.jpg
    :align: center
    :width: 600px

- ``from_skill`` and ``to_skill`` control the skill range of the environment. During initialization, you should match the initial pose of the furniture with the pre-defined pose using GUI tool (see :ref:`Start Teleoperation` list item 3). And then, the script will move the end-effector to the pre-defined pose (plus with noise depending on randomness level) for each skill. Below are the initialization processes of the script when ``from_skill`` is set at 1 to 4, from left to right.

.. |skill1| image:: ../_static/images/skill1.gif
.. |skill2| image:: ../_static/images/skill2.gif
.. |skill3| image:: ../_static/images/skill3.gif
.. |skill4| image:: ../_static/images/skill4.gif

.. table::
    :widths: 25 25 25 25

    +----------+----------+----------+----------+
    | |skill1| | |skill2| | |skill3| | |skill4| |
    +==========+==========+==========+==========+
    |          |          |          |          |
    +----------+----------+----------+----------+


Utilities
---------
The following sections explain the utilities of FurnitureBench.


Visualize Camera Inputs
~~~~~~~~~~~~~~~~~~~~~~~

This script allows you to visualize AprilTag detection and the camera from three different views (front, wrist, and rear):

.. image:: ../_static/images/run_cam_april.png
    :width: 600px

.. code::

    python furniture_bench/scripts/run_cam_april.py


Visualize Robot Trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~~

This script will show robot's trajectory saved in a ``.pkl`` file.
The wrist and front camera views are shown in the left and right panels, respectively.

If you want to try out with the pre-recorded trajectories, you can download the ``.pkl`` files from :ref:`Download dataset`.
We run the following commands with cabinet `trajectory <https://drive.google.com/file/d/1PSh0uvhf7nqFw4KYLf4gn4E7GKferUvD/view?usp=share_link>`__.

.. code::

    python furniture_bench/scripts/show_trajectory.py --data-path 00149.pkl


.. figure:: ../_static/images/trajectory_example.gif
    :align: center
    :width: 80%
    :alt: trajectory_example


Camera Calibration
~~~~~~~~~~~~~~~~~~

Our demonstration consists of randomly perturbed front camera poses in each episode.
To determine the camera pose from the front-view image, we calculate the average camera pose for each type of furniture.

Run the following commands to calibrate the front camera pose for each furniture type.

.. code::

    python furniture_bench/scripts/calibration.py --target <furniture>

.. figure:: ../_static/images/calibration.png
    :width: 60%
    :align: left
    :alt: calibration

    The image displays the deviation of the camera pose from the target pose.
    The green/red text shows if the camera pose is within the threshold or not.
