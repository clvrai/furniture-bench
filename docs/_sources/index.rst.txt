.. FurnitureBench documentation master file, created by
   sphinx-quickstart on Wed Mar 15 17:03:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FurnitureBench
==========================================

FurnitureBench is a real-world furniture assembly benchmark designed to provide a reproducible and user-friendly platform for long-horizon and complex robotic manipulation.
It includes a standardized environment setup and a large dataset, consisting of over 200 hours of data.
Furthermore, we provide a simulator that enables quick testing and development of algorithms through fast iteration.

This document provides technical materials including setup instructions, resources (datasets, 3D models, etc.), tutorials, and API reference for FurnitureBench.
A general overview of the benchmark is available in our `project website <../../../index.html>`_.

.. figure:: _static/images/furniture_teaser.jpg
   :width: 100%
   :alt: furniture_teaser

If you use FurnitureBench in your research, please cite this work as follows:

.. code-block:: bibtex

    @inproceedings{heo2023furniturebench,
      title={FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation},
      author={Minho Heo and Youngwoon Lee and Doohyun Lee and Joseph J. Lim},
      booktitle={Robotics: Science and Systems},
      year={2023}
    }


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/furniturebench_overview.rst
   getting_started/installation_guide.rst
   getting_started/installation_guide_furniture_sim.rst
   getting_started/dataset.rst

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/teleop_oculus_keyboard
   tutorials/furniture_sim
   tutorials/tools_and_scripts.rst

.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   references/code_organization.rst
   references/troubleshooting.rst
   references/development_roadmap.rst
   references/additional_information.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
