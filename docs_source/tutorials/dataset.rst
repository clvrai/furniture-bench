Dataset
=======

Furniture assembly is a complex, long-horizon manipulation task, which is very challenging to solve using reinforcement learning. To make our benchmark tractable, we provide **219.6 hours** of **5100** successful demonstrations collected using an Oculus Quest 2 controller and a keyboard.

Each furniture assembly task has three different levels with respect to the randomness in task initialization: `low`, `medium`, and `high`.


Download Dataset
~~~~~~~~~~~~~~~~

FurnitureBench dataset can be downloaded from `our Google Drive <https://drive.google.com/drive/folders/1j59vFmgBsatu1PZK52HWX_9o5BCh_XDt?usp=sharing>`__.

* Use `rclone <https://rclone.org/>`__ for fast download following `Download with rclone <#download-with-rclone>`__.


Dataset Size
~~~~~~~~~~~~

The size (in GB) of demonstrations for each furniture in each level is summarized below:

+--------------+-----+------+------+
| Furniture    | low | med  | high |
+==============+=====+======+======+
| lamp         | 26  | 27   | 11   |
+--------------+-----+------+------+
| square_table | 76  | 75   | 25   |
+--------------+-----+------+------+
| desk         | 46  | 57   | 25   |
+--------------+-----+------+------+
| drawer       | 43  | 39   | 11   |
+--------------+-----+------+------+
| cabinet      | 38  | 36   | 17   |
+--------------+-----+------+------+
| round_table  | 25  | 26   | 15   |
+--------------+-----+------+------+
| stool        | 37  | 42   | 19   |
+--------------+-----+------+------+
| chair        | 54  | 68   | 31   |
+--------------+-----+------+------+
| one_leg      | 112 | 129  | 69   |
+--------------+-----+------+------+
| Total        | 457 | 499  | 223  |
+--------------+-----+------+------+


Dataset Directory Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

FurnitureBench dataset is structured as follows:
::

   furniture
     |- dataset
       |- furniture_low  # Low randomness
         |- lamp
           |- 0.pkl      # Demonstration 0
           |- 1.pkl
           |- ...
         |- square_table
         |- desk
         |- drawer
         |- cabinet
         |- round_table
         |- stool
         |- chair
         |- one_leg
       |- furniture_med  # Medium randomness
         |- ...
       |- furniture_high # High randomness
         |- ...


Demonstration File Format
~~~~~~~~~~~~~~~~~~~~~~~~~

Each demonstration is stored in a ``.pkl`` file, containing a sequence of sensory inputs, actions, rewards, and other metadata:

::

   'furniture': Furniture name, e.g., 'lamp'
   'observations': List of observation dicts
     {
       'color_image1': Wrist camera image (224, 224, 3)
       'color_image2': Front camera image (224, 224, 3)
       'robot_state': {
         'ee_pos': EEF position (3,)
         'ee_quat': EEF orientation (4,)
         'ee_pos_vel': EEF linear velocity (3,)
         'ee_ori_vel': EEF angular velocity (3,)
         'joint_positions': Joint positions (7,)
         'joint_velocities': Joint velocities (7,)
         'joint_torques': Joint torques (7,)
         'gripper_width': Gripper width (1,)
       }
     }
   'actions': List of 8-D actions
   'rewards': List of rewards (1 if a furniture part is assembled; otherwise, 0)
   'skills': List of skill completion flags (1 if a skill is completed; otherwise, 0)


Download with rclone
~~~~~~~~~~~~~~~~~~~~

1. Install `rclone <https://rclone.org/install/>`__.

2. Run ``rclone config`` to setup Google Drive remote:

.. code::

    No remotes found, make a new one?
    n) New remote
    s) Set configuration password
    q) Quit config
    n/s/q> n
    --------------------

    Enter name for new remote.
    name> furniture
    --------------------

    Choose a number from below, or type in your own value
    Storage> 18
    --------------------

    Two double "Enter" to skip client_id and client_secret
    --------------------

    Choose a number from below, or type in your own value.
    Press Enter to leave empty.
    scope> 2
    --------------------

    Enter a value. Press Enter to leave empty.
    service_account_file> "Enter"
    --------------------

    Edit advanced config?
    y) Yes
    n) No (default)
    y/n> n
    --------------------

    Use web browser to automatically authenticate rclone with remote?
    * Say Y if the machine running rclone has a web browser you can use
    * Say N if running rclone on a (remote) machine without web browser access
    If not sure try Y. If Y failed, try N.

    y) Yes (default)
    n) No
    y/n> n
    --------------------

    Option config_token.
    For this to work, you will need rclone available on a machine that has
    a web browser available.
    For more help and alternate methods see: https://rclone.org/remote_setup/
    Execute the following on the machine with the web browser (same rclone
    version recommended):
            rclone authorize "drive" "<Your config_token>"
    Then paste the result.
    Enter a value.
    config_token>

    *Writer's note*
    # Copy and past `rclone authorize "drive" "<Your config_token>"` in a machine with web browser
    # Login to your Google account
    # Allow rclone to access your Google Drive
    # Past the result to `config_token` in the terminal
    --------------------

    Configure this as a Shared Drive (Team Drive)?

    y) Yes
    n) No (default)
    y/n> n
    --------------------

    Keep this "furniture" remote?
    y) Yes this is OK (default)
    e) Edit this remote
    d) Delete this remote
    y/e/d> y
    --------------------

    Current remotes:

    Name                 Type
    ====                 ====
    furniture            drive

    e) Edit existing remote
    n) New remote
    d) Delete remote
    r) Rename remote
    c) Copy remote
    s) Set configuration password
    q) Quit config
    e/n/d/r/c/s/q> q
    --------------------

3. Connect to Google Drive remote by opening the `dataset Google Drive <https://drive.google.com/drive/u/1/folders/1j59vFmgBsatu1PZK52HWX_9o5BCh_XDt>`__ and clicking "Add a shortcut to Drive", "My Drive", and "Add".

.. image:: ../_static/images/add_shortcut.png
        :width: 400

4. Run the following Python script to download our dataset:

.. code::

    python furniture_bench/scripts/download_dataset.py --randomness [low/med/high] --furniture <name of furniture> --out_dir <path/to/data>

    # E.g., download lamp data with low randomness
    python furniture_bench/scripts/download_data.py --randomness low --furniture lamp --out_dir furniture_dataset

    # E.g., download all furniture data with low randomness
    python furniture_bench/scripts/download_data.py --randomness low --furniture all --out_dir furniture_dataset
