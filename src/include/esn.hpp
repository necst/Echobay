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

#include "EigenConfig.hpp"
#include "IOUtils.hpp"
#include "DataStorage.hpp"
#include "Readout.hpp"

namespace EchoBay
{
    ArrayBO esn_caller(const Eigen::VectorXd &optParams, YAML::Node confParams, 
                       std::string outputFolder, const DataStorage &store, 
                       bool guessEval, const std::string &computationType, 
                       const std::string &matrixFolder);
    
    Reservoir esn_config(const Eigen::VectorXd &optParams, YAML::Node confParams, 
                         int Nu, int guess, const std::string folder);

    ArrayBO esn_train(YAML::Node confParams, Reservoir &ESN, double lambda,
                      std::string problemType, std::string fitnessRule,
                      std::string outputFolder, const DataStorage &store, bool saveflag, 
                      int guesses);

    ArrayBO esn_compute(YAML::Node confParams, Reservoir &ESN, const DataStorage &store, 
                        const std::string &problemType, const std::string &fitnessRule, 
                        int blockStep, Eigen::Ref<MatrixBO> Wout, 
                        const std::string &outputFolder, bool saveflag, int guesses);

    floatBO esn_compute(const MatrixBO &input_data, const std::string &folder);

    MatrixBO compute_prediction(int outNr, int outDimension,
                            const Eigen::Ref<MatrixBO> Wout, 
                            Reservoir &ESN,const MatrixBO &src,
                            const Eigen::Ref<const ArrayBO> sampleState);
} // namespace EchoBay

#endif