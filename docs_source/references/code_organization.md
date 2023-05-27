Code Organization
=========================================================

Here we list the core files and folders in the codebase. The codebase is organized as follows:

```
furniture_bench
  |- __init__.py                          # Register gym environments
  |- assets
    |- franka_description_ros             # Franka Panda URDF, mesh for simulation
    |- furniture                          # Furniture URDF, mesh for simulation
  |- config.py                            # The configuration file for the environment. It contains the configuration for the robot, camera, furniture, and the device interface etc
  |- sim_config.py                        # Configuration file for the simulation environment
  |- controllers
    |- control_utils.py                   # Utility functions for controlling the robot
    |- osc.py                             # Implementation of OSC controller in PyTorch (will run in the `server` workstation)
  |- data
    |- collect_enum.py                    # Define data collection status
    |- data_collector.py                  # Define data collection class that rollout the environment, get action from the interface (e.g., teleoperation, automatic scripts), and save data
  |- envs
    |- policy_envs                        # Environment with different observation spaces for the policy
      |- furniture_bench_image_feature.py # FurnitureBench with image features (R3M or VIP) as observation instead of visual observations
      |- furniture_bench_image.py         # FurnitureBench with 224x224 wrist and front RGB images and robot states as observation
      |- furniture_sim_image_feature.py   # FurnitureSim with image features (R3M or VIP) as observation instead of visual observations
      |- ...
    |- furniture_bench_env.py             # FurnitureBench environment
    |- furniture_sim_env.py               # FurnitureSim environment
    |- initialization_mode.py             # Initialization mode for the environment (e.g., randomness, skill modes)
    |- observation.py                     # Define constant of observation keys
  |- furniture
    |- furniture.py                       # Define base class for all furniture. It contains the core functions and properties for the furniture (e.g., furniture parts, computing reward function, getting observation,etc.)
    |- square_table.py                    # Define square_table furniture
    |- chair.py                           # Define chair furniture
    |- ...                                # Other furniture classes
    |- parts
      |- part.py                          # Define base class for all furniture parts. It contains the core functions and properties for the furniture parts (e.g., reset pose, AprilTag detector, etc.)
      |- cabinet_body.py                  # Define cabinet body part
      |- cabinet_door.py                  # Define cabinet door part
      |- ...                              # Other furniture part classes
  |- device
    |- keyboard_interface.py              # Control the robot using keyboard
    |- oculus_interface.py                # Control the robot using Oculus
    |- device_interface.py                # ABC for device interface
    |- key_enum.py                        # Define key enum for keyboard interface
    |- keyboard_oculus_interface.py       # Control the robot using keyboard and Oculus
    |- __init__.py                        # Utility functions for device interface (e.g., factory function for the device interface)
  |- perception
    |- apriltag.py                        # Define AprilTag detector class
    |- image_utils.py                     # Utility functions for image processing (e.g., image cropping, image resizing used for the policy)
    |- realsense.py                       # Define Realsense camera class
  |- robot
    |- panda.py                           # Panda robot class which build interface to actual robot
    |- robot_state.py                     # Define enum for properties of the robot and robot error. Also define utility functions for the robot states (e.g., flatten robot states.)
  |- scripts
    |- calibration.py                     # Script for extrinsic calibration of front camera
    |- collect_data.py                    # Script for data collection
    |- preprocess_data.py                 # Script for preprocessing the data
    |- reset.py                           # Move the robot to the initial position
    |- run_cam_apriltag.py                # Show camera image and AprilTag detection
    |- show_trajectory.py                 # Show the trajectory of the saved dataset
  |- utils                                # Utility functions for the geometry, camera, robot control etc. (e.g., rotation matrix, quaternion, etc.)
    |- detection.py                       # Define functions for camera reading and AprilTag pose estimation
    |- pose.py                            # Utility for pose (e.g., rotation matrix, quaternion, etc.)
    |- ...
```
