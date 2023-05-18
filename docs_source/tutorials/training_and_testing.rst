Benchmarking FurnitureSim
=========================

This tutorial shows how to train and evaluate a policy on FurnitureSim.

Prerequisites
~~~~~~~~~~~~~

* Install the following packages:

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

* Prepare training data. You can download a dataset (:ref:`Dataset`) or generate it (:ref:`Automated Assembly Script`).


Rollout with Pre-trained Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run our pre-trained policies in FurnitureSim:

.. code::

    cd <path/to/furniture-bench>

    # Policy trained with 1000 scripted demonstrations.
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_iql_r3m_low_sim_1000 --randomness low

The ``test_offline.py`` will read a checkpoint in ``checkpoint/ckpt`` directory from the root of the project.
The directory name of the checkpoint is the same as the ``run_name.seed`` that is used during training (e.g., ``one_leg_full_iql_r3m_low_sim_1000.42`` where one_leg_full_iql_r3m_low_sim_1000 is run name and 42 is the seed).

The real-world rollout is similar to the above command, but you must change the ``--env_name`` corresponding to the real-world environment.

If your ``run_name`` is one of the pre-trained ``run_name`` and currently does not exist under ``checkpoint/ckpt``, the script will download it from Google Drive.

Here is the list of ``run_name`` for the pre-trained policies:

.. code::

    # IQL
    - "one_leg_full_iql_r3m_low_sim_1000" # IQL algorithm trained with 1000 scripted demonstrations in simulation, initialized with low randomness.
    - "one_leg_full_iql_r3m_low_1000"     # IQL algorithm trained with 1000 teleoperated demonstrations in the real world, initialized with low randomness.
    - "one_leg_full_iql_r3m_med_1000"     # IQL algorithm trained with 1000 teleoperated demonstrations in the real world, initialized with medium randomness.
    - "one_leg_full_iql_r3m_mixed_2000"   # IQL algorithm trained with 2000 teleoperated demonstrations in the real world, a combination of low and medium randomness.


Convert Data for Training
~~~~~~~~~~~~~~~~~~~~~~~~~
We provide a tutorial on how to train a policy from scratch using our codebase.
Both for BC and IQL training, you need to convert a raw dataset as follows:

First, convert the data to the format for training:

.. code::

    python furniture_bench/scripts/convert_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo>

    # E.g.
    python furniture_bench/scripts/convert_data.py --in-data-path scripted_sim_demo/one_leg_1000 --out-data-path scripted_sim_demo/one_leg_processed_1000

Train BC
~~~~~~~~



Train IQL
~~~~~~~~~

1) Extract R3M or VIP features from the demonstrations:

.. code::

    python implicit_q_learning/extract_feature.py --furniture <furniture_name> --demo_dir <path/to/data>  --out_file_path <path/to/converted_data> [--use_r3m | --use_vip]

    # E.g.
    python implicit_q_learning/extract_feature.py --furniture one_leg --demo_dir scripted_sim_demo/one_leg_processed/ --out_file_path scripted_sim_demo/one_leg_sim_1000.pkl --use_r3m

2) You can train an IQL policy using the following script. If you want to log using ``wandb``, use these arguments ``--wandb --wandb_entity <entity_name> --wandb_project <project_name>``:

.. code::

    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/<furniture_name> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --data_path=<path/to/pkl> --encoder_type=[vip | r3m]

    # E.g.
    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --data_path=scripted_sim_demo/one_leg_sim_1000.pkl --encoder_type=r3m
