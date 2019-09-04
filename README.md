# EchoBay
EchoBay is a C++, [Eigen](http://eigen.tuxfamily.org) powered library, for the training and deployment of Echo State Networks, with multiple layers and different topologies.
EchoBay employs Limbo library to find the best set of hyper-parameters that maximize a score function.
EchoBay is designed to work almost seamlessly on small computers, microcontrollers (tested on ESP32) and large multi-threaded systems.

## Prerequisites
- [Limbo](http://www.resibots.eu/limbo/) tested on v2.1.0
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) tested on v0.6.2
- [spectra](https://spectralib.org/) tested on v0.7.0

Optional dependencies to read CSV data [fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser)

## Basic Usage
After configuration and building (see docs for details) the basic usage is:

#### Training
```
./echobay train configuration.yml outputFolderName
```
#### Testing
```
./echobay compute modelFolderName
```

## Authors
- [Luca Cerina](https://github.com/LucaCerina)
- [Giuseppe Franco](https://github.com/Giuseppe5)

Other important contributions in this research are from Claudio Gallicchio and Alessio Micheli (Department of Computer Science, University of Pisa) and Marco D. Santambrogio (Dipartimento di Informatica, Elettronica e Bioingegneria, Politecnico di Milano)

### Citing EchoBay
If you use EchoBay in a scientific paper, please cite:

Cerina, Luca, Giuseppe Franco, and Marco D. Santambrogio. (2019) [Lightweight autonomous bayesian optimization of Echo-State Networks.](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-103.pdf), European Symposium on Artificial Neural Networks, ESANN, 2019.