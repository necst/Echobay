# Examples
Here you can find simple examples that covers most of the possibilities of EchoBay.
To generate the dataset use python scripts get_laser_data, get_memory_data and get_wovel_data
All examples can be copied into the EchoBay build folder and executed.
The file template.yml is commented to explain implementation details

### regression
A 1-step prediction problem with SantaFe Laser dataset [1] with a minimal set of optimized hyper-parameters.

### classification
Identification of vowel utterances from Japanese Vowel dataset [3]

### memory
Estimation of the memory capacity [2] of an optimized Echo State Network.

### topology
A 1-step prediction problem with SantaFe Laser dataset [1] with non random topologies.

## References
[1]: Gershenfeld, Neil A., and Andreas S. Weigend. The future of time series. No. XEROX-SPL-93-057. Xerox Corporation, Palo Alto Research Center, 1993.
[2]: Jaeger, Herbert. "The “echo state” approach to analysing and training recurrent neural networks-with an erratum note." Bonn, Germany: German National Research Center for Information Technology GMD Technical Report 148.34 (2001): 13.
[3]: https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels