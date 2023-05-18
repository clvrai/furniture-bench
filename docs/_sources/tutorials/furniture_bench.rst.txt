How to Use FurnitureBench
=========================

FurnitureBench Configuration
----------------------------

FurnitureBench can be configured with the following arguments:

.. code::

    import gym
    import furniture_bench

    env = gym.make(
        'Furniture-Env-v0',
        furniture=...,           # string, [lamp | square_table | desk | drawer | cabinet | round_table | stool | chair | one_leg].
        manual_done=...,         # boolean, if True, the episode ends only when the user presses the 'done' button.
        with_display=...,        # boolean, if True, the environment renders the camera images.
        draw_marker=...,         # boolean, if True, the environment renders the AprilTag marker.
        manual_label=...,        # boolean, if True, allow manual labeling of the reward.
        from_skill=...,          # integer, [0-5) skill index to start from. Note that index `i` denotes the completion of ith skill and commencement of the (i + 1)th skill. For instance, to evaluate the performance of the 5th skill, the index should be set at 4.
        to_skill=...,            # integer, [1-5] skill index to end at. Should be larger than `from_skill`. Expected to perform the full task from `from_skill` onwards, if not specified.
        randomness=...,          # string, [low | med | high], randomness level of the environment.
        high_random_idx=...,     # integer, [0-2], index of the high randomness level. Randomly selected if not specified.
        visualize_init_pose=..., # boolean, if True, visualize the initial pose of furniture parts.
        record_video=...,        # boolean, if True, record the video of the agent's observation
        manual_reset=...,        # boolean, if True, allow a manual reset of the environment.
    )

- ``randomness`` controls the randomness level of the environment.

   - ``low``: The initial pose of each furniture piece is fixed; however, there can be a small human noise during resetting.
   - ``med``: Based on the pre-defined poses in `low`, each part can have translation noise between [-5 cm, 5 cm] and rotational noise between [-45◦, 45◦].
   - ``high``: Furniture parts are randomly initialized on the workspace.

Below is an example of the initialization of the chair, depicted with low, med, and high randomness levels, arranged from left to right.

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

Our demonstration consists of randomly perturbed front camera pose in each episode.
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
