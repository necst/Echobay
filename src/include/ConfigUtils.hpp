#ifndef CONFIGUTILS_HPP
#define CONFIGUTILS_HPP


//#include "Reservoir.hpp" // Not necessary all Reservoir.hpp
#include "EigenConfig.hpp" // Due to Eigen Vector X.
#include "yaml-cpp/yaml.h"

typedef std::map<std::string, Eigen::VectorXd> stringdouble_t;

/**
 * @brief Map a value in the Limbo vector to proper range from the YAML configuration Node
 * 
 * @tparam T type of returned value, int or float
 * @param paramName Name of the parameter in the YAML Node
 * @param layer Layer position in a deep ESN configuration
 * @param extra Number fo extra parameters see TODO add reference to Limbo vector structure
 * @param optParams Eigen Vector used by Limbo to define hyper-parameters
 * @param confParams YAML Node containing hyper-parameters at high level
 * @param defaultValue Default value to be returned in the case that the parameter is not available in the YAML Node
 * @return T Mapped value
 */
template <class T>
T parse_config(const std::string &paramName, const int layer, const int extra, const Eigen::VectorXd &optParams, const YAML::Node confParams, T defaultValue)
{
    T returnValue;
    int idx;
    double upperBound, lowerBound;
    int nDofLayer = confParams["input_dimension_layer"].as<int>(); // Number of optimizable parameters per layer
    int nDofOffset = confParams["input_dimension_general"].as<int>(); // Number of optimizable parameters general
    int index;
    if (confParams[paramName])
    {
        if (confParams[paramName]["type"].as<std::string>() == "dynamic")
        {
            idx = confParams[paramName]["index"].as<int>();

            upperBound = confParams[paramName]["upper_bound"].as<double>();
            lowerBound = confParams[paramName]["lower_bound"].as<double>();
            // Control Off-set
            if (layer == -1)
            {
                index = idx;
            }else{
                index = nDofOffset;
	            // Control extra parameters        
                index += (extra>0) ? (layer * nDofLayer + extra - 1) : (layer * nDofLayer + idx);
            }
            //index = (layer != -1) ? (nDofOffset) : (0);

            // Check output type
            if (std::is_same<T, int>::value)
            {
                returnValue = (int)round(optParams(index) * (upperBound - lowerBound) + lowerBound);
            }
            else
            {
                returnValue = optParams(index) * (upperBound - lowerBound) + lowerBound;
            }
        }
        else //if (confParams[paramName]["type"].as<std::string>() == "fixed")
        {
            returnValue = confParams[paramName]["value"].as<T>();
        }
        return returnValue;
    }
    else
    {
        return defaultValue;
    }
}
#endif