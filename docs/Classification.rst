Advanced example
^^^^^^^^^^^^^^^^
One of the great advantages of ESN is being able to suit different categories of problem, including also classification.

In this particular case, the aim is to provide a label for a input sequence, that can be mono or multi-variate.
In particular each time-step, the ESN will compute a new reservoir state, and at the end of each sequence, whose lengths are defined by the user through a sampling file, the ESN will use the *last* reservoir state as encoding of whole sequence itself. Different approaches are available in literature, each with its pros and cons. 
In our case, we have decided to opt for the one with the lowest computational complexity, keeping in mind a possible embedding implementation of **EchoBay**.

The classification problem is based on the Japanese Vowel Dataset [1]_, commonly used for this types of tasks. The aim is to discriminate between 9 subjects in the pronunciation of the same vowel, exploiting the variation in time of 12 cepstrum coefficients.
More details on the dataset can be found in [1]_.

As for the regression task, also in this case a python script is available to generate the dataset in the appropriate configuration for **EchoBay**, ``get_vowel_data.py``, using:

::

  python3 get_vowel_data.py

The main difference with respect to the regression task is constituted by the presence of three additional file:

* TrainSampling.csv
* ValSampling.csv
* TestSampling.csv

The aim of these files is to tell **EchoBay** at which time-step the reservoir state should be sampled and whether the reservoir state should be reset. 

User-defined sampling
*********************
Both Regression and Classification tasks support user-defined sampling: this option
allow multiple things such as concatenating different streams of data, resetting
the internal state of ESNs and using data windows to discriminate different patterns 
in the data.

A sampling file is a csv composed by two columns. The first column express the 
number of state updates performed for each state sampling. As an example:

::

    1,1,1,1,1,1,1,1,1,1,1

Will sample every state update, while
::

    11,23,31,23,7,34

Will sample the state after 11 updates, then 23, then 31 etc. This is particularly
useful in classification tasks, where we do not care about all the states to train the readout
and where the observed patterns in the data exhibits different lengths.

The second column is either 1 or 0 and tells **EchoBay** to continue updating (1) or 
to reset the internal states to 0 (0) as in Classification tasks. If the washout is larger than 0, they are simply
removed from sampling. At runtime, an additional flag checks if reset is triggered at
the end of training set and manipulate the washout of validation set accordingly.


The sampling file is parsed by **EchoBay** to generate an internal sampling array composed
by 1 (sample data), 0 (avoid sampling), and -1 (reset states).

In the Japanese Vowel case, after each sequence the reservoir state is sampled and then reset. 

The rest of the optimization process is identical to the regression case.


Multi Dataset Handling
**********************

Another possibility offered by EchoBay is to perform the optimization process on multiple dataset using the same *YAML* configuration file.
In order to do this it is required to change the value of ``num_datasets`` in the *YAML* configuration file. To allow **EchoBay** to navigate through all datasets, the name of the folder containing the ``.csv`` files should be in the form of: datafolder1, datafolder2, datafolder3, etc.

In the *YAML* file, the path should be specified in the form of:
::
  train_data: ./datafolder{{x}}/TrainData.csv
  ...

**EchoBay** will take care of replacing *{{x}}* with the appropriate number, and load the correct files.

At the end of the overall process, **EchoBay** will save the results of each optimization procedure in a separate folder.


.. [1] https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels














