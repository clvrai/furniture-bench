Additional Information
======================

3D Printing 🖨️
~~~~~~~~~~~~~~

Here we show an example of sliced the model using FlashPrint:

.. image:: ../_static/images/slice_example.png
    :width: 300px
    :alt: slice_example


Estimated Time to Print
-----------------------
We provide an estimated time for each furniture model in our configuration.
The actual duration may differ based on the type of printer and settings.

+--------------------+--------------------------+
| Furniture Model    | Estimated time to print  |
+====================+==========================+
| lamp               | 12h 47m                  |
+--------------------+--------------------------+
| square_table       | 16h 3m                   |
+--------------------+--------------------------+
| desk               | 22h 39m                  |
+--------------------+--------------------------+
| round_table        | 14h 11m                  |
+--------------------+--------------------------+
| stool              | 9h  53m                  |
+--------------------+--------------------------+
| chair              | 17h 36m                  |
+--------------------+--------------------------+
| drawer             | 23h 55m                  |
+--------------------+--------------------------+
| cabinet            | 18h 47m                  |
+--------------------+--------------------------+


Install Client (CPU-only)
-----------------------------
Here we explain how to install a light-weighted client Docker image, primarily intended for data collection purposes. The image is built upon the ``ubuntu:20.04`` base image.

.. code:: bash

    # Clone the repository and cd into it
    git clone https://github.com/clvrai/furniture-bench.git

    # Pull a pre-built docker image from Docker Hub
    docker pull furniturebench/client:latest

    # Or build the Docker image yourself
    DOCKER_BUILDKIT=1 docker build -t client . -f docker/client.Dockerfile
