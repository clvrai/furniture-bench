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

tmux send-keys -t $session.0 "launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=$ROBOT_IP" ENTER
tmux send-keys -t $session.1 "launch_gripper.py gripper=franka_hand gripper.cfg.robot_ip=$ROBOT_IP" ENTER

tmux a -t $session
