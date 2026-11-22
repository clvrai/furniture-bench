Dataset
=======

Furniture assembly is a complex, long-horizon manipulation task, which is very challenging to solve using reinforcement learning. To make our benchmark tractable, we provide **219.6 hours** of **5100** successful demonstrations collected using an Oculus Quest 2 controller and a keyboard.

Each furniture assembly task has three different levels with respect to the randomness in task initialization: `low`, `medium`, and `high`.


Download Dataset
~~~~~~~~~~~~~~~~

All 5,100 FurnitureBench demonstrations are available in RLDS format through
`Open X-Embodiment (OXE) <https://robotic-transformer-x.github.io/>`__.
Download the `FurnitureBench RLDS files <https://huggingface.co/datasets/lerobot-raw/furniture_bench_dataset_raw/tree/main>`__
from Hugging Face or use Google Cloud Storage.

.. code:: bash

   gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/furniture_bench_dataset_converted_externally_to_rlds .

Demonstration File Format
~~~~~~~~~~~~~~~~~~~~~~~~~

Each demonstration contains sequence of sensory inputs, actions, rewards, and other metadata:

::

   'furniture': Furniture name, e.g., 'lamp'
   'observations': List of observation dicts
     {
       'color_image1': Wrist camera image (224, 224, 3)
       'color_image2': Front camera image (224, 224, 3)
       'robot_state': {
         'ee_pos': EEF position (3,)
         'ee_quat': EEF orientation (4,)
         'ee_pos_vel': EEF linear velocity (3,)
         'ee_ori_vel': EEF angular velocity (3,)
         'joint_positions': Joint positions (7,)
         'joint_velocities': Joint velocities (7,)
         'joint_torques': Joint torques (7,)
         'gripper_width': Gripper width (1,)
       }
     }
   'actions': List of 8-D actions
   'rewards': List of rewards (1 if a furniture part is assembled; otherwise, 0)
   'skills': List of skill completion flags (1 if a skill is completed; otherwise, 0)
