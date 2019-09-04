.. _hyperparams:

Hyperparams basic configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this section there will be a description of all the supported *optimizable* hyper-parameters in **EchoBay**, with a suggestion for *standard* range when optimizing them.

Nr - Reservoir Size
*******************
This parameter refers to the amount of active units present in the Reservoir. It is related to the amount of recall the network has with respect to past input values.
High Nr assures better the performances, but a slower computation.

Typical range of optimization: [300, 1000]

ρ - Spectral Radius
*******************
This parameter allows to rescale the spectral radius (i.e. the largest, in absolute value, of the eigenvalues) of the reservoir weight matrix **Wr**.
This parameter controls, among other things, the recall of the network, and it is strictly tied to the stability of the system itself. Many studies in literature present the stability conditions for ESN. 
A common practice is to set this value to values close to 1, considered the *edge of stability*, although it has been proven that this is not true in practical scenarios.
Using **EchoBay**, there is the possibility to explore values higher than one, considering that in case of instability the performance of the system will fall dramatically, and that particular hyperparameter region will be ignored.

There are two main ways to rescale **Wr**. The *Radius* modality computes the exact spectral radius and perform the rescaling. The *Value* modality simply multiply **Wr** for a rescaling factor according to the work from Gallicchio et al [1]_.

Typical range of optimization: [0.5, 1.3]


σ - Density
***********
This parameter is relevant only for random ESN topology. Small World Topology and Cyclic (Jump) Reservoirs don't require to specify the density.

It controls the sparsity of the reservoir weight matrix **Wr**, and consequently the total number of active units. A higher number of active units does not necessarily imply higher performance.

Typical range of optimization: [0.05, 0.3]


α - Leaky Factor
****************
This parameter influences the "speed" of the reservoir dynamics of the ESN with respect to the input. Small values of α will create a reservoir that reacts slowly to the input signal, thus is more suited to fast-sampled, slow varying signals. 
It can assume values in the range [0,1]. In case of α = 1, the ESN returns to the basic topology.

Typical range of optimization: [0, 1]

ω - Scale Input
***************

These parameters refer to the scaling factors applied to the input matrix **Win**.
Although random initialized, the correct rescaling allows the network to better suit the input dynamics.
In case of multi-variate input signals, the user has the capability of rescaling individually each column of the **Win** matrix, including the one associated to bias. This can be done specifying the *count* field in the ScaleIn parameter, in the YAML file.

Typical range of optimization: [0, 1]



λ - Regularization Factor
*************************
This parameter refers to the regularization factor present in the Ridge Regression solution.

Typical range of optimization: [0, 0.1]


Nl - Number of Layers
*********************
This parameter controls the number of layers in the ESN network. A value of 1 indicates a shallow, basic configuration. In case of values greater than 1, the configuration will correspond to a Deep-ESN. 
In case of Deep-ESN, each layer will have its hyperparameters optimized individually. This means that the more the layers in the ESN, the more the hyperparameters to be optimized.

As of now, it is not possible to optimize Nl through **EchoBay**, so it must be fixed beforehand.

Typical values: between 1 and 10


Washout Samples
***************
This parameter controls how many samples will be discarded as transient at the beginning of the computation of ESN, and, more in general, each time the reservoir state is reset (as may happen in classification problems).
It is not considered to be much relevant in terms of influence on the final result, and for this reason is very often fixed and not optimized.

Typical values: between 200 and 1000, according to the size of the dataset 





.. [1] Gallicchio, Claudio, Alessio Micheli, and Luca Pedrelli. "Fast Spectral Radius Initialization for Recurrent Neural Networks." INNS Big Data and Deep Learning conference. Springer, Cham, 2019.