Troubleshooting
===============

.. |br| raw:: html

  <br/>

`Polymetis <https://facebookresearch.github.io/fairo/polymetis/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Q:** What should I do if I get the ``communication_constraints_violation`` error on the server while using Polymetis (Library for Franka robot interface)?

  **A:** Consider disabling CPU frequency scaling. Refer to this page for `instructions <https://frankaemika.github.io/docs/troubleshooting.html#disabling-cpu-frequency-scaling>`__

**Q:** What should I do if I encounter ``OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory`` error?

  **A:** If you encounter a Segmentation fault (core dumped) error in the client PC while using a robot, you should rebuild fairo by running the command below in the Docker image. This might be caused by the linking error because of the reinstall of PyTorch.

  .. code::

     $ cd /fairo/polymetis/polymetis/build/
     $ make -j

**Q:** I'm encountering issues while building Polymetis from source. How can I resolve this?


    **A:** Here are some common errors and solutions.

    - For ``/home/linuxbrew/.linuxbrew/Cellar/openssl@1.1/1.1.1t/lib/libcrypto.so: undefined reference to `dlsym@GLIBC_2.34'`` error:

        Follow the steps below.

        .. code::

            # 1. Unlink the brew openssl (https://github.com/Homebrew/homebrew-core/issues/118825)
            brew unlink openssl@1.1

            # 2. build the Polymetis again.
            cd <path/to/fairo>/polymetis/polymetis/

            # Remove the build directory if it exists.
            rm -rf build
            mkdir build && cd build
            # Rebuild.
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF
            make -j

    - For ``Failed to detect a default CUDA architecture.`` error:

        Make sure the CUDA path is set correctly.

        .. code::

            # E.g.,
            export PATH=/usr/local/cuda-11.7/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH


        Ensure you can run the following command.

        .. code::

            nvcc -V

        You should see something like this.

        .. code::

            nvcc: NVIDIA (R) Cuda compiler driver
            Copyright (c) 2005-2022 NVIDIA Corporation
            Built on Wed_Jun__8_16:49:14_PDT_2022
            Cuda compilation tools, release 11.7, V11.7.99
            Build cuda_11.7.r11.7/compiler.31442593_0

    - For ``/home/user/fairo/polymetis/polymetis/torch_isolation/include/torch_server_ops.hpp:56:39: error: ‘size_t’ has not been declared`` error:

        Add ``#include <stddef.h>`` on top of the ``torch_server_ops.hpp`` file, and build again.

        1) Open the file.

        .. code::

            vim <path/to/fairo>/polymetis/polymetis/torch_isolation/include/torch_server_ops.hpp

        2) And add the following line on top of the file.

        .. code::

            #include <stddef.h>

**Q:** What should I do with warnings like
``Warning: Failed to load 'libtorchrot.so' from CONDA_PREFIX`` or
``Warning: Failed to load 'libtorchscript_pinocchio.so' from CONDA_PREFIX``


  **A:** It does not affect the functionality of the system, so you can ignore it.

Device Connections
~~~~~~~~~~~~~~~~~~

.. note::
    Make sure all the devices (cameras, Oculus if you are collecting the data) are using USB 3.x.

| **Q:** How can I check if my devices (cameras, Oculus) are using USB 3.x?

  **A:** Run ``lsusb`` and ``lsusb -t``. When you run ``lsusb -t``,
  the communication speed in Mbps located at the end of each line must be
  equal to or above 5000M (USB 3.0).

  For example,

  .. code:: bash

     $ lsusb
     Bus 002 Device 006: ID 8086:0b07 Intel Corp. Intel(R) RealSense(TM) Depth Camera 435
     Bus 002 Device 007: ID 8086:0b07 Intel Corp. Intel(R) RealSense(TM) Depth Camera 435

     Bus 004 Device 008: ID 2833:0183 GenesysLogic USB3.2 Hub
     Bus 004 Device 002: ID 05e3:0625 Genesys Logic, Inc. USB3.2 Hub
     Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub

     $ lsusb -t
     /:  Bus 04.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/2p, 10000M
         |__ Port 2: Dev 2, If 0, Class=Hub, Driver=hub/4p, 5000M
             |__ Port 2: Dev 8, If 0, Class=Imaging, Driver=usbfs, 5000M
             |__ Port 2: Dev 8, If 1, Class=Vendor Specific Class, Driver=, 5000M
             |__ Port 2: Dev 8, If 2, Class=Vendor Specific Class, Driver=usbfs, 5000M
     /:  Bus 02.Port 1: Dev 1, Class=root_hub, Driver=xhci_hcd/6p, 5000M
         |__ Port 2: Dev 7, If 0, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 2: Dev 7, If 1, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 2: Dev 7, If 2, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 2: Dev 7, If 3, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 2: Dev 7, If 4, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 5: Dev 6, If 4, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 5: Dev 6, If 2, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 5: Dev 6, If 0, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 5: Dev 6, If 3, Class=Video, Driver=uvcvideo, 5000M
         |__ Port 5: Dev 6, If 1, Class=Video, Driver=uvcvideo, 5000M


| **Q:** The robot does not follow Oculus Quest 2 even after the connection is established. What should I do?

  **A:** Please check the following:

    - Make sure you find Oculus device when running `adb devices` commands in Client.
    - Please double-check if you follow the instructions in the :ref:`Setup Oculus Quest 2` section.
    - If the problem persist, restart the Oculus.

Training and Evaluation
~~~~~~~~

| **Q:** What should I do if I face a CUDA Out of memory (OOM) issue while trying to learn implicit_q_learning (IQL)?

  **A:** If you face a CUDA Out of memory (OOM) issue while trying to learn implicit_q_learning (IQL), it might be due to preallocated GPU memory. You can adjust the memory fraction by setting ``XLA_PYTHON_CLIENT_MEM_FRACTION=.XX`` to resolve this issue.

| **Q:** What should I do if I face ``Access denied with the following error:`` or `FileNotFoundError: [Errno 2] No such file or directory: '/root/.r3m/r3m_50/model.pt'` while downloading r3m checkpoints?

  **A:** This might be due to the permission issue. Please download the checkpoints manually from the Google Drive and copy them to the Docker image.

  - (Here we show the example of downloading the checkpoint for ``r3m ResNet50``.)
  - Download the `checkpoint <https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA>`__ in your local machine
  - Get the container ID by running ``docker ps``
  - Copy the checkpoint to the container by running ``docker cp <checkpoint_path> <container_id>:/root/.r3m/r3m_50/``


Oculus
~~~~~~

| **Q:** What should I do to prevent sudden actions from the robot due to wrong signal readings when using Oculus?

  **A:** To prevent sudden actions from the robot due to wrong signal readings when using Oculus, ensure that the cable connection is stable.

| **Q:** What if the robot is not moving when I use Oculus?

  **A:** Make sure to control the robot in the guidance area of Oculus, allow the access to the Oculus, and verify that the device is visible and accessible by running adb devices. Also check the Oculus is turned on (white light is on in the front).

Camera
~~~~~~

| **Q:** How can I check if my camera is connected stably?

  **A:** Consider
  installing `realsense
  viewer <https://robots.uc3m.es/installation-guides/install-realsense2.html>`__
  and test whether the camera is connected stably. Also, there are other
  features in the viewer that can be used to check the camera status.

| **Q:** What should I do if I encounter a RuntimeError: Frame didn't arrive within 5000 error when using a camera?

  **A:** You should unplug
  your camera and then plug it back in.

| **Q:** What does the error message “RuntimeError: xioctl(VIDIOC_S_FMT) failed Last Error: Device or resource busy” mean when working with a camera?

  **A:** This error message indicates that there is another
  program, such as realsense-viewer or a Python code, using the camera.
  The camera should only run in a single program at a time. To resolve
  this issue, check if there is another program that may be using the
  camera and close it before running the desired program.

  .. note::

      - Make sure recent firmware is installed. (Our setting was 05.13.00.50 version)
      - Make the camera is connected using USB 3.x

Simulator
~~~~~~~~~

| **Q:** What should I do if I encounter an error ``isaacgymenvs setup command: 'python_requires' must be a string containing valid version specifiers; Invalid specifier: '>=3.6.*`` during local installation?

  **A:** execute the following commands, and then rerun the installation.

  .. code:: bash

    pip install --upgrade pip wheel
    pip install setuptools==58
    pip install --upgrade pip==22.2.2

| **Q:** I am encountering ``ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`` error.

  **A:** Run following commands.

    .. code::

        sudo apt update
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt update
        sudo apt install python3.8-dev

| **Q:** What should I do if I encounter an error ``[Error] [carb.windowing-glfw.plugin] GLFW initialization failed.`` or ``No protocol specified`` while running in Docker image?

  **A:** Shut down the current Docker container, run ``xhost +``, and then restart the Docker container.

| **Q:** What should I do if I encounter an error ``[Error] [carb.gym.plugin] cudaExternamMemoryGetMappedBuffer failed on rgbImage buffer wit h error 101``?

  **A:** You should specify vulkan explicitly.

  Shut down the current Docker container, and then run the following commands

  .. code:: bash

    apt install vulkan-tools
    MESA_VK_DEVICE_SELECT=list vulkaninfo

  Rerun the Docker container, and then specify device

  .. code:: bash

    # e.g.,
    MESA_VK_DEVICE_SELECT='10de:2204' python furniture_bench/scripts/run_sim_env.py --furniture square_table --no-action


| **Q:** Simulator does not terminate even after I press Ctrl+C. What should I do?

  **A:** It could happen when the input streams are blocked. The workaround is to press Ctrl+Z and then ``kill %1`` to terminate the first job.

Gym
~~~

| **Q:** What should I do if I encounter an observation space error while working with Gym? (such as ``'python_requires' must be a string containing valid version specifiers; Invalid specifier: '>=3.6.*'``)

  **A:** Install Gym version 0.21.0 by running
  ``pip install gym==0.21.0``.

| **Q:** I am getting the error while running ``pip install gym==0.21.0`` or ``pip install -r requirements.txt``

  **A:** Run the following commands

  .. code::

    pip install --upgrade pip wheel
    pip install setuptools==58
    pip install --upgrade pip==22.2.2

Display
~~~~~~~

| **Q:** What should I do if I encounter an error ``: cannot connect to X server :1``?

  **A:** Shut down the current Docker container, run ``xhost +``, and then restart the Docker container.
