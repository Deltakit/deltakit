.. title:: Deltakit Documentation

Deltakit
========

Deltakit integrates quantum error correction circuits generation,
advanced circuit simulation, software decoders and noise analysis tools into
quantum error correction (QEC) experiment workflows. Deltakit offers
offline decoding capabilities that are complementary to Riverlane's
real-time hardware decoders, with more flexibility and functionality
than currently implemented in hardware. By running small experiments
on today’s quantum hardware and simulators, users gain an understanding of how error
correction procedures will perform on future large scale quantum computers.
This allows users to study QEC algorithms and protocol, to showcase QEC experiments on
current quantum computers, as well as diagnose noise sources to further
improve the logical performance of their devices. Deltakit
can be used to perform simulations of all parts of QEC experiments
if a QPU is not available.

With Deltakit, you can:

* **Benchmark logical fidelity** by decoding measurement data from QEC experiments or simulations
* **Decode QEC experiments and simulations** using open-source decoders, as well as Riverlane proprietary decoders
* **Improve decoding accuracy** by using measurement data to determine decoding graph
* **Diagnose noise sources** from QEC experiments by analysing defect rates and correlation matrices

To get started with Deltakit, follow these steps:

* See the :doc:`Setup <setup>` section to find out how to receive a user account, access the software and set up your working environment;
* See the :doc:`Getting Started <../guide/getting_started>` guide for an overview of available features;
* See the :doc:`Examples <../examples/notebooks/simulation/stim_simulation>` section for end-to-end demonstrations of how to use the available features;
* See the :doc:`API Reference <../api>` section for detailed information on how to access the available features.

.. list-table::
   :widths: 50 50
   :class: borderless

   * - .. figure:: _static/images/codes.png
          :alt: Error-correcting codes
          :align: center
          :width: 250px
          :target: https://deltakit.readthedocs.io/en/latest/api.html#deltakit-explorer-codes

     - .. figure:: _static/images/qpu_noise.png
          :alt: QPU and noise analysis
          :align: center
          :width: 250px
          :target: https://deltakit.readthedocs.io/en/latest/api.html#deltakit-explorer-qpu

   * - .. figure:: _static/images/experiments.png
          :alt: Experiments
          :align: center
          :width: 250px
          :target: https://deltakit.readthedocs.io/en/latest/api.html#deltakit-explorer

     - .. figure:: _static/images/decoders.png
          :alt: Decoders
          :align: center
          :width: 250px
          :target: https://deltakit.readthedocs.io/en/latest/api.html#deltakit-decode

.. toctree::
   :maxdepth: 2
   :hidden:

   Home <self>
   setup
   guide/getting_started
   examples/index

.. toctree::
   :maxdepth: 1
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :hidden:

   CONTRIBUTING
