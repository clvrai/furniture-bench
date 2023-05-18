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

The pre-trained policies for FurnitureBench can be evaluated similar to FurnitureSim. You only need to change ``--env_name`` to the one for the real-world environment.


Convert Data for Training
~~~~~~~~~~~~~~~~~~~~~~~~~

Both for BC and IQL training, you need to convert a raw dataset as follows:

.. code::

    python furniture_bench/scripts/convert_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo>

    # E.g.,
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

    # E.g.,
    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --data_path=scripted_sim_demo/one_leg_sim_1000.pkl --encoder_type=r3m
