Training and Testing
=========================

This tutorial shows how to train and evaluate a policy on FurnitureBench and FurnitureSim.


Prerequisites
~~~~~~~~~~~~~

* Run following commands to install the dependencies for training and testing.

  .. code::

    cd <path/to/furniture-bench>
    ./install_model_deps.sh

* Now, install JAX with GPU support based on your CUDA version. See `official doc <https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier>`__.

* Make sure your PyTorch and JAX are installed properly. You can run the following commands to check the installation.

  .. code::

    python -c "import torch; print(f'PyTorch installed successfully, with GPU support = {torch.cuda.is_available()}')"
    python -c "import jax.numpy as jnp; jnp.ones((3,)); print('JAX installed successfully')"

* Prepare training data. You can download the FurnitureBench dataset (:ref:`Dataset`) or generate one in FurnitureSim (:ref:`Automated Assembly Script`).


Evaluating Pre-trained Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section shows how to evaluate pre-trained policies of BC and Implicit Q-Learning (IQL) algorithms.

Evaluating Pre-trained IQL
--------------------------

You can run our pre-trained IQL policies in FurnitureSim using ``implicit_q_learning/test_offline.py``.

.. code::

    cd <path/to/furniture-bench>

    python implicit_q_learning/test_offline.py --env_name=FurnitureSimImageFeature-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_iql_r3m_low_sim_1000 --randomness low

* If you use the pair of ``run_name`` and ``seed`` that we provide, the pre-trained checkpoint will be automatically downloaded from Google Drive. The checkpoint will be saved in ``checkpoint/ckpt/<run_name>.<seed>`` (e.g., ``one_leg_full_iql_r3m_low_sim_1000.42`` for run name ``one_leg_full_iql_r3m_low_sim_1000`` and seed ``42``).

* The below table shows the list of pre-trained ``run_name`` and ``seed``:

==============================================          ====================================================================================
              Run name / seed                                  Note
==============================================          ====================================================================================
``one_leg_full_iql_r3m_low_sim_1000`` / ``42``          IQL trained with 1000 scripted demos in simulation, low randomness.
``one_leg_full_iql_r3m_low_1000``     / ``42``          IQL trained with 1000 real-world demos, low randomness.
``one_leg_full_iql_r3m_med_1000``     / ``42``          IQL trained with 1000 real-world demos, medium randomness.
``one_leg_full_iql_r3m_mixed_2000``   / ``42``          IQL trained with 2000 real-world demos, low and medium randomness.
==============================================          ====================================================================================

* To evaluate the real-world policies, you must change ``--env_name`` with the real-world environment: ``FurnitureBenchImageFeature-v0``.


Evaluating Pre-trained BC
-------------------------
BC policies are evaluated using ``run.py``.

.. code::

    python -m run env.id=FurnitureSim-v0 env.furniture=one_leg run_prefix=<run_prefix> init_ckpt_path=<path/to/checkpoint> rolf.encoder_type=<encoder_type> is_train=False gpu=<gpu_id> env.randomness=<randomness>

    # E.g., evaluate a pre-trained BC policy with ResNet18 encoder
    python -m run env.id=FurnitureSim-v0 env.furniture=one_leg run_prefix=one_leg_full_bc_resnet18_low_sim_1000 init_ckpt_path=checkpoints/ckpt/one_leg_full_bc_resnet18_low_sim_1000/ckpt_00000000050.pt rolf.encoder_type=resnet18 is_train=False gpu=0 env.randomness=low

* To evaluate the real-world policies, set ``env.name=FurnitureBenchImage-v0``.

Training a Policy from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a tutorial on how to train a policy from scratch using our codebase.


Preprocess Data for Training
-------------------------

Both for BC and IQL training, you need to convert a raw dataset as follows:

.. code::

    python furniture_bench/scripts/preprocess_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo>

    # E.g., convert data in `scripted_sim_demo/one_leg` and store in `scripted_sim_demo/one_leg_processed`
    python furniture_bench/scripts/preprocess_data.py --in-data-path scripted_sim_demo/one_leg --out-data-path scripted_sim_demo/one_leg_processed

To extract skill-specific segmented trajectories, use ``--from-skill`` and ``--to-skill``:

.. code::

    python furniture_bench/scripts/preprocess_data.py --in-data-path <path/to/demos> --out-data-path <path/to/processed/demo> --from-skill <skill_index> --to-skill <skill_index>

Training BC
-----------
The following command trains a BC policy. You can change ``rolf.encoder_type`` to ``resnet18``, ``resnet32``, ``resnet50``, ``r3m``, or ``vip``. If you want to log using ``wandb``, use these arguments: ``wandb=True wandb_entity=<entity_name> wandb_project=<project_name>``.

.. code::

    python -m run run_prefix=<run_prefix> rolf.demo_path=<path/to/processed/demo> env.furniture=<furniture> rolf.encoder_type=<encoder_type> gpu=<gpu_id>

    # E.g., train BC with ResNet18 encoder
    python -m run run_prefix=one_leg_full_bc_resnet18_low_sim rolf.demo_path=scripted_sim_demo/one_leg_processed/ env.furniture=one_leg rolf.encoder_type=resnet18 rolf.finetune_encoder=True gpu=0

The setup for BC training is specified in the file ``rolf/rolf/config/algo/bc.yaml``. This configuration will be merged with the default settings for the training. The merged configuration will be stored in the ``config`` directory, following the naming convention: ``FurnitureDummy-v0.bc.<run_prefix>.<seed>.yaml``.

Evaluating BC
-------------

To evaluate a BC policy, add ``is_train=False`` and the checkpoint path to evalute ``init_ckpt_path=log/FurnitureDummy-v0.bc.<run_prefix>.<seed>/ckpt/<checkpoint name>``.

.. code::

    python -m run env.id=FurnitureSim-v0  run_prefix=<run_prefix> env.furniture=<furniture> rolf.encoder_type=<encoder_type> gpu=<gpu_id> is_train=False init_ckpt_path=<path/to/checkpoint>

    # E.g., evaluate BC with ResNet18 encoder
    python -m run env.id=FurnitureSim-v0  run_prefix=one_leg_full_bc_resnet18_low_sim env.furniture=one_leg rolf.encoder_type=resnet18 gpu=0 is_train=False init_ckpt_path=log/FurnitureDummy-v0.bc.one_leg_full_bc_resnet18_low_sim.123/ckpt/ckpt_00000000050.pt


Training IQL
------------

1) Extract R3M or VIP features from the demonstrations:

.. code::

    python implicit_q_learning/extract_feature.py --furniture <furniture> --demo_dir <path/to/data> --out_file_path <path/to/converted_data> [--use_r3m | --use_vip]

    # E.g., extract R3M features from the dataset
    python implicit_q_learning/extract_feature.py --furniture one_leg --demo_dir scripted_sim_demo/one_leg_processed/ --out_file_path scripted_sim_demo/one_leg_sim.pkl --use_r3m

2) You can train an IQL policy using the following script. If you want to log using ``wandb``, use these arguments: ``--wandb --wandb_entity <entity_name> --wandb_project <project_name>``.

.. code::

    python implicit_q_learning/train_offline.py --env_name=FurnitureImageFeatureDummy-v0/<furniture> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --data_path=<path/to/pkl> --encoder_type=[vip | r3m]

    # E.g., train IQL with R3M features
    python implicit_q_learning/train_offline.py --env_name=FurnitureImageFeatureDummy-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --data_path=scripted_sim_demo/one_leg_sim.pkl --encoder_type=r3m


Evaluating IQL
--------------

To evaluate an IQL policy, run ``implicit_q_learning/test_offline.py`` as follows:

.. code::

    export XLA_PYTHON_CLIENT_PREALLOCATE=false

    python implicit_q_learning/test_offline.py --env_name=FurnitureSimImageFeature-v0/<furniture> --config=implicit_q_learning/configs/furniture_config.py --run_name <run_name> --encoder_type=[vip | r3m] --ckpt_step <ckpt_step>

    # E.g., evaluate IQL with R3M features
    python implicit_q_learning/test_offline.py --env_name=FurnitureSimImageFeature-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --run_name one_leg_sim --encoder_type=r3m --ckpt_step 1000000
