How to use EchoBay
^^^^^^^^^^^^^^^^^^
The **EchoBay** library exploits the Limbo optimization library to search the best set 
of hyper-parameters with a Bayesian Optimization rule.

The Limbo library stores all the parameters to be optimized in a compact way
using a Eigen Vector. All the values are in the range [0, 1].

In order to simplify the process to ESN experts and data science researchers, the
optimization ranges and other useful variables are contained in a configuration file
structured with YAML (see yaml.org for infos)

Get EchoBay
***********
To get **EchoBay**, clone the source code from https://github.com/LucaCerina/Echobay
with git, or download it as a zip.

Dependencies
************
The **EchoBay** library require some mandatory dependencies to function properly and
some extra dependencies that could be used to increase performance.

Mandatory dependencies are:

- Limbo_ (tested on v2.1.0)
- yaml-cpp_ (tested on v0.6.2)
- spectra_ (tested on v0.7.0)

Both Eigen3_ and (in minimal part) Boost_ are also dependencies of the Limbo library
and are required by **EchoBay**. It is better to install Eigen3 from latest repository rather than using Ubuntu PPA.

YAML library yaml-cpp should be compiled with -DBUILD_SHARED_LIBS=ON.

Additional libraries that could be used to parallelize the calculation of Echo States
and initial random sampling are:

- OpenMP_ (v4.5): boost Eigen calculations.
- Intel TBB_ (also a Limbo optional dependency): allow parallel sampling.


.. _Limbo: http://www.resibots.eu/limbo/
.. _yaml-cpp: https://github.com/jbeder/yaml-cpp
.. _spectra: https://spectralib.org/
.. _Eigen3: http://eigen.tuxfamily.org
.. _Boost: https://www.boost.org/
.. _OpenMP: https://www.openmp.org
.. _TBB: https://www.threadingbuildingblocks.org/

Installation and testing
************************
The following guide was tested on a x86 machine with a fresh Ubuntu 16.04 Virtual Machine.
After the installation of the dependencies, the steps required are:

Build Limbo with:
::

./waf configure
./waf build

Then:

.. - Copy the wscript file in the src/limbo folder where Limbo is installed $LIMBO_FOLDER

- Copy src/waf_tools folder in $LIMBO_FOLDER

- Create a exp/EchoBay folder in $LIMBO_FOLDER

- Finally, copy the content of src folder in $LIMBO_FOLDER/exp/EchoBay

Build **EchoBay** with:
::

./waf configure --exp Echobay
./waf build --exp EchoBay

The result will be the build/exp/EchoBay folder. To test the correct functioning, you can copy
the files in the examples folder and run it with:
::

./echobay train configfile.yml outputfoldername

After the optimization process, the results will be in *outputfolder*. Other than
Limbo output (see the Limbo docs for details), you will find the optimal matrices
in sparse market format, the output computed in the testing phase, and another YAML
file containing the optimal configuration.

Using external fitness functions
********************************
The **EchoBay** library supports also external regression fitness functions with pybind and Python3.6.
Classification functions currently require label encoding which is left to the final user.
see :ref:`Comparator` for details.

This extension require other dependencies, specifically Python headers and pybind11.
Python headers can be installed with:
::

  apt-get install python3-dev

While pybind11_ should be installed from its repo.

.. _pybind11: https://github.com/pybind/pybind11

After the installation, you can copy src/waf_tools/pybind.py in $LIMBO_FOLDER/waf_tools
and reconfigure the library with:
::

./waf configure --exp EchoBay --pybind

Now you can call your external function changing the configuration file as:

.. code-block:: cpp

   Problem_Definition : {type: "External", Fitness_Function: "filename"}

Where *filename* is the name of python file containing the fitness function. See *fitnessfunctions* folder for a template.
An example is the Mean Absolute Error:

.. code-block:: python
   
    import numpy as np
    # All functions should be named fitness and receive predicted data and true labels
    def fitness(predict, actual):
        samples = predict.shape[0]
        mae = np.sum(np.abs(predict - actual))/samples

        return mae