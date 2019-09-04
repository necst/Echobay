Classes
-------
The **EchoBay** Library is built using multiple modular classes under the same namespace 
that could be used together in the Limbo Optimization framework or separately 
to add Echo State Network functionalities to your project.

Reservoir
^^^^^^^^^
.. doxygenclass:: EchoBay::Reservoir
   :members:

DataStorage
^^^^^^^^^^^
.. doxygenclass:: EchoBay::DataStorage
   :members:

Comparator
^^^^^^^^^^
.. doxygenclass:: EchoBay::Comparator
   :members:

EchoBay namespace
-----------------
ComputeState
^^^^^^^^^^^^
These functions update the non-linear states of Echo State Networks.

.. doxygenfile:: ComputeState.hpp

ESN
^^^

.. doxygenfile:: esn.hpp

Utilities
---------
Other headers are used for all the methods related to ESN management, like Eigen configuration,
data IO, configuration utilities, CLI printing

EigenConfig
^^^^^^^^^^^
Modify this file to change floating point precision of the entire Library

.. doxygenfile:: EigenConfig.hpp

LimboParams
^^^^^^^^^^^
Structures related to Limbo Bayesian Optimization

.. doxygenfile:: LimboParams.hpp

IOUtils
^^^^^^^
Manage input and output from disk

.. doxygenfile:: IOUtils.hpp

ConfigUtils
^^^^^^^^^^^
Map parameters in the YAML configuration to values in the Limbo vector

.. doxygenfile:: ConfigUtils.hpp


