#!/bin/bash

# Check ROBOT_IP environment variable is set
if [ -z "$ROBOT_IP" ]; then
    echo "ROBOT_IP is not set"
    exit 1
fi

session="server"
tmux has-session -t $session 2>/dev/null
if [ $? != 0 ]; then
	tmux new-session -d -s server
	tmux split-window -v
fi

# Check if --FR3 argument is given and set the robot client accordingly
if [[ "$1" == "--FR3" ]]; then
    robot_client="franka_FR3_hardware"
    robot_model="franka_research3"
else
    robot_client="franka_hardware"
    robot_model="franka_panda"
fi

tmux send-keys -t $session.0 "launch_robot.py robot_client=$robot_client robot_model=$robot_model robot_client.executable_cfg.robot_ip=$ROBOT_IP" ENTER

# Launch the gripper configuration based on --FR3 argument
if [[ "$1" == "--FR3" ]]; then
    tmux send-keys -t $session.1 "launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=$ROBOT_IP" ENTER
else
    tmux send-keys -t $session.1 "launch_gripper.py gripper=franka_hand gripper.cfg.robot_ip=$ROBOT_IP" ENTER
fi

tmux attach -t $session
