#ifndef ESN_HPP
#define ESN_HPP

#include <iostream>
#include <boost/filesystem.hpp>

#ifdef USE_TBB
#include <tbb/mutex.h>
#endif

#include "yaml-cpp/yaml.h"

#include "Reservoir.hpp"
#include "FitnessFunctions.hpp"
#include "ComputeState.hpp"

#include "EchoBay.hpp"
#include "IOUtils.hpp"
#include "DataStorage.hpp"
#include "Readout.hpp"

namespace EchoBay
{
    ArrayBO esn_caller(const Eigen::VectorXd &optParams, YAML::Node confParams, 
                       const std::string outputFolder, const DataStorage &store, 
                       const bool guessEval, const std::string &computationType, 
                       const std::string &matrixFolder);
    
    Reservoir esn_config(const Eigen::VectorXd &optParams, YAML::Node confParams, 
                         const int Nu, const int guess, const std::string folder);

    ArrayBO esn_train(YAML::Node confParams, Reservoir &ESN, const double lambda,
                      const std::string problemType, const std::string fitnessRule,
                      const std::string outputFolder, const DataStorage &store, const bool saveflag, 
                      const int guesses);

    ArrayBO esn_compute(YAML::Node confParams, Reservoir &ESN, const DataStorage &store, 
                        const std::string &problemType, const std::string &fitnessRule, 
                        const int blockStep, Eigen::Ref<MatrixBO> Wout, 
                        const std::string &outputFolder, const bool saveflag, const int guesses);

    floatBO esn_compute(const MatrixBO &input_data, const std::string &folder);

    MatrixBO compute_prediction(const int outNr, const int outDimension,
                            const Eigen::Ref<MatrixBO> Wout, 
                            Reservoir &ESN, const MatrixBO &src,
                            const Eigen::Ref<const ArrayBO> sampleState);
} // namespace EchoBay

#endif