Benchmarking FurnitureSim
=========================

This tutorial shows how to train and evaluate a policy on FurnitureSim.


Prerequisites
~~~~~~~~~~~~~

* Install the packages for benchmarking:

  .. code::

    cd <path/to/furniture-bench>
    pip install -e rolf
    pip install -e r3m
    pip install -e vip
    pip install -r implicit_q_learning/requirements.txt


* Prepare training data. You can download a dataset (:ref:`Dataset`) or generate it (:ref:`Automated Assembly Script`).


Evaluating Pre-trained Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section shows how to evaluate pre-trained policies of BC and Implicit Q-Learning (IQL) algorithms.
The examples provided below utilize our previously trained checkpoints, but it will be straightforward to modify certain parameters and adapt them to your own policies.

Evaluating Pre-trained IQL
----------------------------------

You can run our pre-trained IQL policies in FurnitureSim using ``implicit_q_learning/test_offline.py``.

.. code::

    cd <path/to/furniture-bench>

    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_iql_r3m_low_sim_1000 --randomness low


- If you use one of the pre-trained ``run_name``, the pre-trained checkpoint will be automatically downloaded from Google Drive. The checkpoint will be saved in ``checkpoint/ckpt/<run_name>.<seed>`` (e.g., ``one_leg_full_iql_r3m_low_sim_1000.42`` for run name ``one_leg_full_iql_r3m_low_sim_1000`` and seed ``42``).

- The below table shows the list of pre-trained ``run_name``:

===================================== ====================================================================================
              Run name                         Note
===================================== ====================================================================================
``one_leg_full_iql_r3m_low_sim_1000`` IQL trained with 1000 scripted demos in simulation, low randomness.
``one_leg_full_iql_r3m_low_1000``     IQL trained with 1000 real-world demos, low randomness.
``one_leg_full_iql_r3m_med_1000``     IQL trained with 1000 real-world demos, medium randomness.
``one_leg_full_iql_r3m_mixed_2000``   IQL trained with 2000 real-world demos, a combination of low and medium randomness.
===================================== ====================================================================================

- To evaluate the real-world policies, you must change ``--env_name`` with the real-world environment.

Evaluating Pre-trained BC
----------------------------------
BC policies are evaluated using ``run.py``.

.. code::

    # Run the following command to evaluate a BC policy.
    python -m run algo@rolf=bc env.id=Furniture-Image-Sim-Env-v0 env.furniture=one_leg init_ckpt_path=<path/to/checkpoint> rolf.encoder_type=<encoder_type> is_train=False gpu=<gpu_id> rolf.resnet=<resnet_type> env.randomness=<randomness>
    # E.g., pre-train BC with ResNet18 encoder.
    python -m run algo@rolf=bc env.id=Furniture-Image-Sim-Env-v0 env.furniture=one_leg init_ckpt_path=checkpoints/ckpt/one_leg_full_bc_resnet18_low_sim_1000/ckpt_00000000050.pt rolf.encoder_type=resnet18 is_train=False gpu=0 rolf.resnet=resnet18 env.randomness=low


Training a Policy from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a tutorial on how to train a policy from scratch using our codebase.


Convert Data for Training
-------------------------

Both for BC and IQL training, you need to convert a raw dataset as follows:

.. code::

    python furniture_bench/scripts/convert_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo>

    # E.g.,
    python furniture_bench/scripts/convert_data.py --in-data-path scripted_sim_demo/one_leg_1000 --out-data-path scripted_sim_demo/one_leg_processed_1000


Training BC
--------
The following command trains a BC policy. You can change ``rolf.encoder_type`` to ``resnet18``, ``r3m``, or ``vip``.

.. code::

    python -m run run_prefix=<run_prefix> algo@rolf=bc env.id=Furniture-Image-Dummy-v0 rolf.demo_path=<path/to/processed/demo> env.furniture=<furniture> rolf.encoder_type=<encoder_type> rolf.resnet=<resnet_type> rolf.finetune_encoder=True gpu=<gpu_id> wandb=[True | False]  wandb_entity=<wandb_entity> wandb_project=<wandb_project>

    # E.g., train BC with ResNet18 encoder.
    python -m run run_prefix=one_leg_full_bc_resnet18_low_sim_1000 algo@rolf=bc env.id=Furniture-Image-Dummy-v0 rolf.demo_path=one_leg_processed_1000/ env.furniture=one_leg rolf.encoder_type=resnet18 rolf.resnet=resnet18 rolf.finetune_encoder=True wandb=True gpu=0 wandb_entity=clvr wandb_project=furniture-bench


Training IQL
---------

1) Extract R3M or VIP features from the demonstrations:

.. code::

    python implicit_q_learning/extract_feature.py --furniture <furniture> --demo_dir <path/to/data>  --out_file_path <path/to/converted_data> [--use_r3m | --use_vip]

    # E.g.,
    python implicit_q_learning/extract_feature.py --furniture one_leg --demo_dir scripted_sim_demo/one_leg_processed/ --out_file_path scripted_sim_demo/one_leg_sim_1000.pkl --use_r3m

2) You can train an IQL policy using the following script. If you want to log using ``wandb``, use these arguments ``--wandb --wandb_entity <entity_name> --wandb_project <project_name>``:

.. code::

    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/<furniture> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --data_path=<path/to/pkl> --encoder_type=[vip | r3m]

    # E.g.,
    python implicit_q_learning/train_offline.py --env_name=Furniture-Image-Feature-Dummy-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --data_path=scripted_sim_demo/one_leg_sim_1000.pkl --encoder_type=r3m
