Training and Testing
==================

This tutorial

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

Rollouts with Pre-trained Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    cd <path/to/furniture-bench>
    python implicit_q_learning/test_offline.py --env_name=Furniture-IQL-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_r3m_1000 --randomness low



Policy Training and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate trained policy:

Training the policy with IQL:

.. code::

    # BC conversion
    python furniture_bench/scripts/convert_data.py --in-data-path /data/minho/hdd/IL_data/one_leg/ --out-data-path /data/minho/converted_one_leg_mixed_2000/

    # IQL conversion
    python implicit_q_learning/convert_furniture_data.py --furniture one_leg --demo_dir /hdd/converted_stool_full_100 --out_file_path one_leg_sim.pkl --use_r3m
TODO
