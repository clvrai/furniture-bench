#!/bin/bash

# Set default options
# Parse the first argument.

# Check the first agument is low or med or high, otherwise exit.
if [ "$1" = "low" ]; then
    echo "Running low randomness"
elif [ "$1" = "med" ]; then
    echo "Running medium randomness"
elif [ "$1" = "high" ]; then
    echo "Running high randomness"
else
    echo "Unknown first argument: $1"
    exit 1
fi

# one_leg
if [ "$2" = "one_leg" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/one_leg --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name one_leg_full_r3m_1000 --randomness $1
# square_table
elif [ "$2" = "square_table" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/square_table --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name square_table_full_r3m_1000 --randomness $1
# desk
elif [ "$2" = "desk" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/desk --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name desk_full_r3m_1000 --randomness $1
# chair
elif [ "$2" = "chair" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/chair --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name chair_full_r3m_1000 --randomness $1
# round_table
elif [ "$2" = "round_table" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/round_table --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name round_table_full_r3m_1000 --randomness $1
# lamp
elif [ "$2" = "lamp" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/lamp --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name lamp_full_r3m_1000 --randomness $1
# cabinet
elif [ "$2" = "cabinet" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/cabinet --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name cabinet_full_r3m_1000 --randomness $1
# drawer
elif [ "$2" = "drawer" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/drawer --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name drawer_full_r3m_1000 --randomness $1
# stool
elif [ "$2" = "stool" ]; then
    python implicit_q_learning/test_offline.py --env_name=Furniture-Image-Feature-Sim-v0/stool --config=implicit_q_learning/configs/furniture_config.py --ckpt_step=1000000 --run_name stool_full_r3m_1000 --randomness $1
# Raise
else
    echo "Unknown second argument: $2"
    exit 1
fi
