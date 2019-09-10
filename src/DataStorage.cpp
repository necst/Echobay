#include "DataStorage.hpp"

using namespace Eigen;

/**
 * @brief Load data from files
 * 
 * @param dataFile path to the data file
 * @param labelFile path to the label file
 * @param type differentiate training data ("train") or validation/test ("valid")
 */
void EchoBay::DataStorage::load_data(const std::string dataFile, const std::string labelFile, const std::string type)
{
    if (type == "train")
    {
        load_csv(dataFile, _trainData);
        load_csv(labelFile, _trainLabel);
    }
    else // If it is not training it is validation/testing
    {
        load_csv(dataFile, _evalData);
        load_csv(labelFile, _evalLabel);
    }
}

/**
 * @brief Copy data from existing matrices
 * 
 * @param data Eigen Matrix of data
 * @param label Eigen Matrix of labels
 * @param type differentiate training data ("train") or validation/test ("valid")
 */
void EchoBay::DataStorage::copy_data(Eigen::Ref<MatrixBO> data, Eigen::Ref<MatrixBO> label, const std::string type)
{
    if (type == "train")
    {
        _trainData = data;
        _trainLabel = label;
    }
    else // If it is not training it is validation/testing
    {
        _evalData = data;
        _evalLabel = label;
    }
}

/**
 * @brief Return data or labels based on parameters
 * 
 * @param type train or validation
 * @param select data or label
 * @return MatrixBO Selection 
 */
MatrixBO EchoBay::DataStorage::get_data(const std::string type, const std::string select)
{
    if (type == "train")
    {
        if (select == "data")
        {
            return _trainData;
        }
        else
        {
            return _trainLabel;
        }
    } 
    else // If it is not training it is validation/testing
    {
        if (select == "data")
        {
            return _evalData;
        }
        else
        {
            return _evalLabel;
        }
    }
}

/**
 * @brief Return sampling array
 * 
 * @param type Train or validation
 * @return ArrayBO Output sampling Array
 */
ArrayBO EchoBay::DataStorage::get_sampleArray(const std::string type)
{
    if(type == "train")
    {
        return _trainSampling;
    }
    else
    {
        return _evalSampling;
    } 
}

/**
 * @brief Normalize sampling vector to be used in other functions
 * 
 * @param samplingData Eigen Matrix with sampling format TODO explain this
 * @param nWashout Number of washout samples
 * @param init_flag Flag that controls if washout resets also ESN states
 * @param problemType Classification or Regression/Memory
 * @param type train or validation
 * @return ArrayBO Cleaned sampling array
 */
ArrayBO EchoBay::DataStorage::set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const std::string type)
{
    int nSamples = (type == "train") ? _trainData.rows() : _evalData.rows();
    // Generate sampling vector
    ArrayBO resetSamples(samplingData.rows());
    ArrayBO samplingPoints;
    // Place 1 to sample data
    samplingPoints = (problemType == "Classification") ? ArrayBO::Constant(nSamples, 0) : ArrayBO::Constant(nSamples, 1);

    // Get cumulative sum and differences
    std::partial_sum(samplingData.col(0).data(), samplingData.col(0).data() + samplingData.col(0).size(), resetSamples.data());

    // TODO Check washout size
    // if(nWashout >= SamplingTrain.col(0).minCoeff() || nWashout >= SamplingVal.col(0).minCoeff())
    // {
    //     std::cout << "Washout is oversize with respect to smallest data window" << std::endl;
    //     throw std::exception();
    // }

    // Initial washout
    // Keep in consideration if reset was triggered at the end of TrainingSet
    if (init_flag)
    {
        samplingPoints.block(0, 0, nWashout, 1) = ArrayBO::Constant(nWashout, 0);
    }
    // Other resets
    for (int s = 0; s < resetSamples.rows(); s++)
    {
        if (samplingData(s, 1) == 0)
        {
            // place -1 to reset state
            samplingPoints[resetSamples[s] - 1] = -1;
            for (int washout = resetSamples[s]; (washout < (resetSamples[s] + nWashout)) && (washout < nSamples); washout++)
            {   
                // Place 0 to not sample information
                samplingPoints[washout] = 0;
            }
        }
        else if(samplingData(s,1) == 1)
        {
            samplingPoints[resetSamples[s] - 1] = 1;
        }
    }

    if(type == "train")
    {
        _trainSampling = samplingPoints;
    }
    else
    {
        _evalSampling = samplingPoints;
    }

    return samplingPoints;
}