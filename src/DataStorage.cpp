#include "DataStorage.hpp"

using namespace Eigen;

/**
 * @brief Load data from files
 * 
 * @param dataFile path to the data file
 * @param labelFile path to the label file
 * @param type differentiate training data ("train") or validation/test ("valid")
 */
void EchoBay::DataStorage::load_data(const std::string dataFile, const std::string labelFile, const uint8_t type)
{
    uint8_t _type = (type == EchoBay::Train) ? 0 : 1; // Avoid bad calls
    load_csv(dataFile, _seriesData[_type]);
    load_csv(labelFile, _seriesLabel[_type]);
}

/**
 * @brief Copy data from existing matrices
 * 
 * @param data Eigen Matrix of data
 * @param label Eigen Matrix of labels
 * @param type differentiate training data ("train") or validation/test ("valid")
 */
void EchoBay::DataStorage::copy_data(Eigen::Ref<MatrixBO> data, Eigen::Ref<MatrixBO> label, const uint8_t type)
{
    uint8_t _type = (type == EchoBay::Train) ? 0 : 1; // Avoid bad calls
    _seriesData[_type] = data;
    _seriesLabel[_type] = label;
}

/**
 * @brief Return data or labels based on parameters
 * 
 * @param type train or validation
 * @param select data or label
 * @return MatrixBO Selection 
 */
MatrixBO EchoBay::DataStorage::get_data(const uint8_t type, const uint8_t select) const
{
    uint8_t _type = (type == EchoBay::Train) ? 0 : 1; // Avoid bad calls
    return (select == EchoBay::selData) ? _seriesData[_type] : _seriesLabel[_type];
}

/**
 * @brief Get columns of _trainData or _evalData
 * 
 * @param type train or validation
 * @return int number of columns
 */
int EchoBay::DataStorage::get_dataCols(const uint8_t type) const
{
    return (type == EchoBay::Train) ? _seriesData[0].cols() : _seriesData[1].cols();
}

/**
 * @brief Get length (rows) of _trainData or _evalData
 * 
 * @param type train or validation
 * @return int number of samples
 */
int EchoBay::DataStorage::get_dataLength(const uint8_t type) const
{
    return (type == EchoBay::Train) ? _seriesData[0].rows() : _seriesData[1].rows();
}

/**
 * @brief Return sampling array
 * 
 * @param type Train or validation
 * @return ArrayBO Output sampling Array
 */
ArrayI8 EchoBay::DataStorage::get_sampleArray(const uint8_t type) const
{
    return (type == EchoBay::Train) ? _samplingFull[0] : _samplingFull[1];
}

/**
 * @brief Getter function for internal sampling vector
 * 
 * @return std::vector<ArrayBO> Internal sampling vector
 */
std::vector<ArrayI8> EchoBay::DataStorage::get_samplingBatches(const uint8_t type) const
{
    return (type == EchoBay::Train) ? _samplingBatches[0] : _samplingBatches[1];
}

/**
 * @brief Get linear offset to access data using batches
 * 
 * @param type Train or validation
 * @param batch 0-indexed batch position
 * @return int Starting offset for that batch
 */
int EchoBay::DataStorage::get_dataOffset(const uint8_t type, const int batch) const
{
    return (type == EchoBay::Train) ? _dataOffset[0][batch] : _dataOffset[1][batch];
}

/**
 * @brief get number of sampled state across all batches
 * 
 * @param type Train or validation
 * @return int number of sampled states
 */
int EchoBay::DataStorage::get_maxSamples(const uint8_t type, const int batch) const
{
    auto sumSamples = [](int sum, const ArrayI8 curr) { return sum + curr.count(); };
    int idx = (type == EchoBay::Train) ? 0 : 1;
    auto finalBatch = (batch == -1) ? _samplingBatches[idx].end() : _samplingBatches[idx].begin() + batch;
    int counter = std::accumulate(_samplingBatches[idx].begin(), finalBatch, 0, sumSamples);

    return counter;
}

/**
 * @brief Calculate the number of unique batches in samplingData
 * 
 * @param samplingData Eigen Matrix with sampling format
 * @return int Number of unique batches
 */
int EchoBay::DataStorage::get_nBatches(Eigen::Ref<MatrixBO> samplingData)
{
    int nBatches = (samplingData.col(1).array() == 0).count();
    nBatches = nBatches <= 1 ? nBatches + 1 : nBatches;
    return nBatches;
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
ArrayI8 EchoBay::DataStorage::set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const uint8_t type)
{
    int nSamples = _seriesData[type].rows(); //(type == EchoBay::Train) ? _trainData.rows() : _evalData.rows();
    int nBatches = get_nBatches(samplingData);
    // Generate sampling vectors
    ArrayI resetSamples(samplingData.rows());
    // Index that controls activity on training vector or validation vector
    int arrIdx = (int)type; //(type == "train") ? 0 : 1;
    // Clear old vectors TODO this should be improved
    _samplingBatches[arrIdx].clear();
    _samplingBatches[arrIdx].reserve(nBatches);
    _dataOffset[arrIdx].clear();
    _dataOffset[arrIdx].reserve(nBatches);

    ArrayI8 samplingPoints;

    // Get cumulative sum and differences
    auto sumIntCast = [](floatBO x, floatBO y){return (int)(x+y);};
    std::partial_sum(samplingData.col(0).data(), samplingData.col(0).data() + samplingData.col(0).size(), resetSamples.data(), sumIntCast);

    // Place 1 to sample data
    samplingPoints = (problemType == "Classification") ? ArrayI8::Constant(resetSamples.tail(1)[0], 0) : ArrayI8::Constant(resetSamples.tail(1)[0], 1);
    // TODO Check washout size
    // if(nWashout >= SamplingTrain.col(0).minCoeff() || nWashout >= SamplingVal.col(0).minCoeff())
    // {
    //     std::cout << "Washout is oversize with respect to smallest data window" << std::endl;
    //     throw std::exception();
    // }
    if (nBatches == 1)
    {
        ArrayI8 support = (problemType == "Classification") ? ArrayI8::Constant(nSamples, 0) : ArrayI8::Constant(nSamples, 1);
        // Push to sampler
        _samplingBatches[arrIdx].push_back(support);
        _dataOffset[arrIdx].push_back(0);
    }
    else
    {
        int lastReset = 0;
        std::vector<Eigen::Index> idxSupport;
        for (int s = 0; s < resetSamples.rows(); s++)
        {
            if ((samplingData(s, 1) == 0) | (s == resetSamples.rows() - 1))
            {
                ArrayI8 support = (problemType == "Classification") ? ArrayI8::Constant(resetSamples[s] - lastReset, 0) : ArrayI8::Constant(resetSamples[s] - lastReset, 1);
                // Annotate last reset
                lastReset = resetSamples[s];
                // Add washout
                if (nWashout > 0)
                {
                    support.head(nWashout) << ArrayI8::Zero(nWashout);
                }
                // Add partial samples
                for (const auto &sample : idxSupport)
                {
                    support[sample] = 1;
                }
                idxSupport.clear();
                // Add reset
                support.tail(1) << -1;
                // Push back data offset
                if (_samplingBatches[arrIdx].size() < 1)
                {
                    _dataOffset[arrIdx].push_back(0);
                }
                else
                {
                    _dataOffset[arrIdx].push_back(_dataOffset[arrIdx].back() + _samplingBatches[arrIdx].back().rows());
                }
                // Push to sampler
                _samplingBatches[arrIdx].push_back(support);
            }
            else if (samplingData(s, 1) == 1)
            {
                // Accumulate samples an then assign them at the next reset
                int idx = resetSamples[s] > 0 ? resetSamples[s] - 1 : resetSamples[s];
                idxSupport.push_back(idx - lastReset);
            }
        }
    }

    // Initial washout
    // Keep in consideration if reset was triggered at the end of TrainingSet
    // TODO this should be not necessary now
    if (init_flag)
    {
        samplingPoints.block(0, 0, nWashout, 1) = ArrayI8::Constant(nWashout, 0);
    }
    int startSample = 0;
    for (const auto &batch : _samplingBatches[arrIdx])
    {
        samplingPoints.block(startSample, 0, batch.rows(), 1) << batch;
        startSample += batch.rows();
    }
    _samplingFull[type] = samplingPoints;

    return samplingPoints;
}