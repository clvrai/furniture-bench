Training and Testing
====================

This tutorial shows the process for training and evaluating a policy for furniture assembly tasks.
It is presumed that you have already prepared your dataset, whether it was downloaded from Google Drive, collected by yourself, or generated with the scripted agent in simulation.

Prerequisites
~~~~~~~~~~~~~
Install the following packages:

.. code::

    cd <path/to/furniture-bench>

    # Install robot learning library
    cd rolf
    pip install -e .

    # Install implicit Q learning.
    cd implicit_q_learning
    pip install -r requirements.txt

    # Install R3M.
    cd ../r3m
    pip install -e .

    # Install VIP
    cd ../vip
    pip install -e .

Rollout with Pre-trained Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here, we show how to roll out pre-trained policies in simulation.
You can run the following command:

.. code::

    cd <path/to/furniture-bench>

    # Policy trained with 1000 scripted demonstrations.
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_iql_r3m_low_sim_1000 --randomness low

The ``test_offline.py`` will read a checkpoint in ``checkpoint/ckpt`` directory from the root of the project.
The directory name of the checkpoint is the same as the ``run_name.seed`` that is used during training (e.g., ``one_leg_full_iql_r3m_low_sim_1000.42`` where one_leg_full_iql_r3m_low_sim_1000 is run name and 42 is the seed).

If your ``run_name`` is one of the pre-trained ``run_name`` and currently does not exist under ``checkpoint/ckpt``, the script will download it from Google Drive.

Here is the list of ``run_name`` for the pre-trained policies:

.. code::

    # IQL
    - "one_leg_full_iql_r3m_low_sim_1000" # IQL algorithm trained with 1000 scripted demonstrations in simulation, initialized with low randomness.
    - "one_leg_full_iql_r3m_low_1000"     # IQL algorithm trained with 1000 teleoperated demonstrations in the real world, initialized with low randomness.
    - "one_leg_full_iql_r3m_med_1000"     # IQL algorithm trained with 1000 teleoperated demonstrations in the real world, initialized with medium randomness.
    - "one_leg_full_iql_r3m_mixed_2000"   # IQL algorithm trained with 2000 teleoperated demonstrations in the real world, combination of low and medium randomness.

The real-world rollout is similar to the above command, but you must change the ``--env_name`` corresponding to the real-world environment.


BC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, convert the data to the format for training:

.. code::

    python furniture_bench/scripts/convert_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo>

    # E.g.
    python furniture_bench/scripts/convert_data.py --in-data-path scripted_sim_demo/one_leg_1000 --out-data-path scripted_sim_demo/one_leg_processed_1000

IQL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) Extract R3M or VIP features from the demonstrations:

.. code::

    python implicit_q_learning/extract_feature.py --furniture <furniture_name> --demo_dir  --out_file_path <path/to/the/pkl> --<use_r3m or use_vip>
    # E.g.
    python implicit_q_learning/extract_feature.py --furniture one_leg --demo_dir scripted_sim_demo/one_leg_processed/ --out_file_path scripted_sim_demo/one_leg_sim_1000.pkl --use_r3m

Note that ``demo_dir`` should be the directory where the ``converted`` demonstrations with ``convert_data.py``

2) Train an IQL policy

.. code::

    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/<furniture_name> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --data_path=<path/to/pkl> --encoder_type=<vip or r3m>
    # E.g.
    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --data_path=scripted_sim_demo/one_leg_sim_1000.pkl --encoder_type=r3m

    # To use wandb
    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/<furniture_name> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --data_path=<path/to/pkl> --encoder_type=<vip or r3m> --wandb --wandb_entity <entity_name> --wandb_project <project_name>
