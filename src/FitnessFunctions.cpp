#include "FitnessFunctions.hpp"
#include <iostream>
#include <iterator>

// Mutex Lock for python functions
#ifdef USE_TBB
#ifndef GIL_MU
tbb::mutex gil_mutex;
#endif
#endif

/**
 * @brief Construct a Comparator class for a given problem with a specific fitness function
 * 
 * @param problemType Classification, Regression or MemoryCapacity
 * @param fitnessRule fitness function dependent from the chosen problemType
 */
EchoBay::Comparator::Comparator(const std::string &problemType, const std::string &fitnessRule)
{
    // Assign internal variables
    _problemType = problemType;
    _fitnessRule = fitnessRule;
    try
    {
        switch(problemTypes.at(problemType))
        {
            // Regression
            case 0:
                if (fitnessRule == "MSA")
                {
                    fitnessFunction = MSA;
                }
                else
                {
                    fitnessFunction = NRMSE;
                }
                break;
            // Classification
            case 1:
                if (fitnessRule == "F1Mean")
                {
                    fitnessFunction = F1Mean;
                }
                else
                {
                    fitnessFunction = Accuracy;
                }
                break;
            // Memory capacity
            case 3:
                fitnessFunction = MemoryCapacity;
                break;
            // External function
#ifdef USE_PYBIND
            case 4:
                fitnessFunction = ExtFitness;
                break;
#endif
        }
    } 
    catch (const std::exception &e)
    {
        std::cout << "Undefined problemType" << std::endl;
    }
}

/**
 * @brief Resize the internal outputLabel according to problemType
 * 
 * @param rows Rows of the outputLabel
 * @param cols Columns of the outputLabel, multiplied by nClasses in Classification problem 
 * (see also get_targetMatrix(const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayBO> sampleState))
 */
void EchoBay::Comparator::set_label_size(int rows, int cols)
{
    // Copy internal variables
    _rows = rows;
    _cols = cols;

    // Resize matrix
    if (_problemType == "Classification")
    {
        _nOutput = _nClasses;
        _outputLabel.resize(_rows, _cols * _nClasses);
    }
    else // Regression and Memory Capacity
    {
        // TODO needs a check if _outCols is defined or not
        _nOutput = _outCols;
        _outputLabel.resize(_rows, _outCols);
    }
}

/**
 * @brief Evaluate fitness function against internal targetLabel
 * 
 * see also set_targetLabel(const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayBO> sampleState) 
 * for details on internal targetLabel
 * 
 * @param predict Input matrix with predicted values
 * @return floatBO Value of the fitness function evaluation
 */
floatBO EchoBay::Comparator::get_fitness(Eigen::Ref<MatrixBO> predict)
{
    return fitnessFunction(predict, _targetLabel, _nOutput, _outputLabel, _fitnessRule);
}

/**
 * @brief Return the number of classes in a classification label vector
 * 
 * @param label Eigen Matrix containing the different classes
 * @return int Number of unique classes found in the label Matrix
 */
int EchoBay::Comparator::get_nClasses(const Eigen::Ref<const MatrixBO> label)
{
    // Copy label
    ArrayBO _label = label.col(0).array();
    // Get number of classes
    // Sort vector
    std::sort(_label.data(), _label.data() + _label.size(), std::less<floatBO>());
    // Get uniques
    _nClasses = std::distance(_label.data(), std::unique(_label.data(), _label.data() + _label.size()));
    return _nClasses;
}

/**
 * @brief Return the internal nClasses variable
 * 
 * @return int Number of unique classes in the label
 */
int EchoBay::Comparator::get_nClasses()
{
    return _nClasses;
}

/**
 * @brief Set internal targetLabel according to Comparator problemType
 * 
 * Classification problems use the same label and then calculate the number of classes.
 * Regression and MemoryCapacity problems resize targetLabel and manage washout samples
 * 
 * @param label Eigen Matrix containing the desired target label
 * @param sampleState Eigen Array containing the sampling index to manage ESN washout
 * see also EchoBay::DataStorage::set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const std::string type)
 */
void EchoBay::Comparator::set_targetLabel(const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayBO> sampleState)
{
    if (_problemType == "Classification")
    {
        _targetLabel = label;
        _nClasses = get_nClasses(_targetLabel);
    }
    else // Regression and Memory Capacity
    {
        // Count columns of output
        _outCols = label.cols();

        // Copy data in realOut
        _targetLabel.resize(sampleState.count(), _outCols);
        int outRows = 0;
        for (int s = 0; s < sampleState.rows(); s++)
        {
            if (sampleState(s) != 0)
            {
                _targetLabel.row(outRows) = label.row(s);
                outRows++;
            }
        }
    }
}

/**
 * @brief Transform the training targetMatrix according to the problemType
 * 
 * Classification problems use the same label and then perform one-hot encoding
 * Regression and MemoryCapacity problems resize targetLabel and manage washout samples
 * 
 * @param label Eigen Matrix containing the desired target label
 * @param sampleState Eigen Array containing the sampling index to manage ESN washout
 * see also EchoBay::DataStorage::set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const std::string type)
 * @return MatrixBO Problem dependent target matrix used for readout calculation
 * see also MatrixBO EchoBay::Wout_ridge(int rows, int cols, double lambda, Eigen::Ref<MatrixBO> biasedState, Eigen::Ref<MatrixBO> target)
 */
MatrixBO EchoBay::Comparator::get_targetMatrix(const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayBO> sampleState)
{
    if (_problemType == "Classification")
    {
        // realOut equals label in classification
        // Count classes
        get_nClasses(label);

        // Resize targetMatrix
        _targetMatrix.resize(label.rows(), _nClasses);

        // Assign labels
        one_hot_encoding(_targetMatrix, label);
    }
    else // Regression and Memory Capacity
    {
        // Count columns of output
        _outCols = label.cols();

        // Copy data in realOut
        MatrixBO realOut(sampleState.count(), _outCols);
        int outRows = 0;
        for (int s = 0; s < sampleState.rows(); s++)
        {
            if (sampleState(s) != 0)
            {
                realOut.row(outRows) = label.row(s);
                outRows++;
            }
        }

        // Resize targetMatrix
        _targetMatrix.resize(realOut.rows(), _outCols);
        _targetMatrix.setZero(); // TODO Is this necessary
        // Assign label
        _targetMatrix = realOut;
    }

    return _targetMatrix;
}

/**
 * @brief Find the median of an std::vector of floating-point numbers
 * 
 * @param input Vector passed to the function
 * @return floatBO Median value
 */
floatBO EchoBay::Comparator::find_median(std::vector<floatBO> input)
{
    if (input.size() < 1)
        return std::numeric_limits<floatBO>::signaling_NaN();
    if (!(input.size() % 2))
    {
        // Find the two middle positions
        const auto i1 = input.begin() + (input.size()) / 2 - 1;
        const auto i2 = input.begin() + input.size() / 2;

        // Partial sort
        std::nth_element(input.begin(), i1, input.end());
        const auto e1 = *i1;
        std::nth_element(input.begin(), i2, input.end());
        const auto e2 = *i2;

        return (e1 + e2) / 2;
    }
    else
    {
        // Find median on odd arrays
        const auto median_it = input.begin() + input.size() / 2;
        std::nth_element(input.begin(), median_it, input.end());
        return *median_it;
    }

    return input[input.size() / 2];
}

/**
 * @brief Find the median of an Eigen Array of floating-point numbers
 * 
 * @param input Array passed to the function
 * @return floatBO Median value
 */
floatBO EchoBay::Comparator::find_median(Eigen::Ref<ArrayBO> input)
{
    if (input.size() < 1)
        return std::numeric_limits<floatBO>::signaling_NaN();
    if (!(input.size() % 2))
    {
        // Find the two middle positions
        const auto i1 = input.data() + (input.size()) / 2 - 1;
        const auto i2 = input.data() + input.size() / 2;

        // Partial sort
        std::nth_element(input.data(), i1, input.data() + input.size());
        const auto e1 = *i1;
        std::nth_element(input.data(), i2, input.data() + input.size());
        const auto e2 = *i2;

        return (e1 + e2) / 2;
    }
    else
    {
        // Find median on odd arrays
        const auto median_it = input.data() + input.size() / 2;
        std::nth_element(input.data(), median_it, input.data() + input.size());
        return *median_it;
    }

    return input[input.size() / 2];
}

/**
 * @brief Transform linear labeling in one-hot encoding format
 * 
 * @param Dst Encoding destination matrix
 * @param Src Input source label
 */
void EchoBay::Comparator::one_hot_encoding(Eigen::Ref<MatrixBO> Dst, MatrixBO Src)
{
    int k;
    int len = Src.rows();

    // Reset matrix
    Dst.setZero();
    // Assign encoding
    for (int i = 0; i < len; ++i)
    {
        k = (int)Src(i);
        Dst(i, k) = 1;
    }
}


void EchoBay::Comparator::multi_out_encoding(Eigen::Ref<MatrixBO> input_data,
                                             Eigen::Ref<MatrixBO> target_data)
{
    // init data
    int rows = target_data.rows();
    int cols = target_data.cols();
    for (int i = 0; i < rows; ++i)
    {
        int stride_jump = i * cols;
        for (int j = 0; j < cols; ++j)
        {
            target_data(i, j) = input_data(stride_jump + j, 0);
        }
    }
}

/**
 * @brief Estimate confusion matrix starting from prediction and ground truth
 * 
 * @param predict Eigen Matrix containing predicted labels
 * @param actual Eigen Matrix containing actual labels
 * @param nOutput number of classes
 * @param ConfusionMat Output confusion matrix TODO make this return value
 */
void EchoBay::Comparator::ConfusionMatrix(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<Eigen::MatrixXi> ConfusionMat)
{
    int i, c;
    for (size_t j = 0; j < (size_t)actual.size(); j++)
    {
        c = (int)actual(j, 0);
        for (i = 0; i < nOutput; i++)
        {
            ConfusionMat(c, i) += (int)predict(j, 0) == i;
        }
    }
}

/**
 * @brief Calculate fitness function as Median Symmetric accuracy
 * 
 * @f[ MSA = e^{median(|\log(\frac{predict}{actual}|)} - 1 @f]
 * 
 * @param predict Eigen matrix of predicted values
 * @param actual Eigen matrix of actual values
 * @param nOutput Number of output columns
 * @param OutputLabel [deprecated] return the outputLabel 
 * @return floatBO Value of the fitness function
 */
floatBO EchoBay::Comparator::MSA(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    // Variables
    int i;
    ArrayBO ratio_array(predict.rows());
    ArrayBO MSA_vals(predict.cols());
    floatBO output;

    // Calculate MSA
    for (i = 0; i < nOutput; i++)
    {
        ratio_array = ((predict.col(i).array()) / (actual.col(i).array() + 1e-15)).log().abs(); // TODO TO BE FIXED SOMEHOW
        MSA_vals(i) = expf(find_median(ratio_array)) - 1;
    }

    output = 100.0 * (1.0 - MSA_vals.mean());
    output = output > 0 ? output : 0.0;
    return output;
}

/**
 * @brief Calculate fitness function as Normalized Root-Mean Square Error
 * 
 * @f[ NRMSE = \frac{\sqrt{\sum_{i=0}^N (predict_i - actual_i)}}{\sqrt{\sigma^2_{actual}}} @f]
 * 
 * @param predict Eigen matrix of predicted values
 * @param actual Eigen matrix of actual values
 * @param nOutput Number of output columns
 * @param OutputLabel [deprecated] return the outputLabel
 * @param rule Unused. Necessary to be compliant with external fitness functions 
 * @return floatBO Value of the fitness function
 */
floatBO EchoBay::Comparator::NRMSE(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    // Variables
    int i;
    ArrayBO diff_array(predict.rows());
    floatBO RMSE, variance;
    floatBO var_correction;
    ArrayBO NRMSE(predict.cols());

    // Calculate NRMSE
    var_correction = predict.rows() / (floatBO)(predict.rows() - 1);
    for (i = 0; i < nOutput; i++)
    {
        // Calculate difference array
        diff_array = predict.col(i) - actual.col(i);

        RMSE = sqrt(diff_array.pow(2).mean());
        variance = var_correction * (actual.col(i).array().pow(2).mean() - pow(actual.col(i).array().mean(), 2));
        NRMSE(i) = RMSE / sqrt(variance);
    }

    return 100.0 * (1.0 - NRMSE.mean());
}

/**
 * @brief Calculate fitness function as F1 classification accuracy
 * 
 * 
 * @param predict Eigen matrix of predicted values
 * @param actual Eigen matrix of actual values
 * @param nOutput Number of output classe
 * @param OutputLabel [deprecated] return the outputLabel
 * @param rule Unused. Necessary to be compliant with external fitness functions 
 * @return floatBO Value of the fitness function
 */
floatBO EchoBay::Comparator::F1Mean(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    // This part can be put in a function as well
    MatrixBO label(actual.rows(), actual.cols());
    int argMax;

    for (size_t i = 0; i < (size_t)actual.size(); ++i)
    {
        //Compute Argmax to determine the class
        predict.row(i).maxCoeff(&argMax);
        label(i, 0) = ((floatBO)argMax);
    }
    // Compute Confusion Matrix
    Eigen::MatrixXi ConfusionMat(nOutput, nOutput);

    ConfusionMat.setZero();
    ConfusionMatrix(label, actual, nOutput, ConfusionMat);
    std::cout << "\n" << ConfusionMat << std::endl;

    floatBO F1 = 0;
    floatBO RowWise = 0;
    floatBO ColWise = 0;
    floatBO actualClasses = 0;
    floatBO F1temp = 0;

    for (int i = 0; i < nOutput; i++)
    {
        RowWise = (floatBO)(ConfusionMat.row(i).sum());
        ColWise = (floatBO)(ConfusionMat.col(i).sum());
        if (RowWise > 0)
        {
            F1temp = 2 * ConfusionMat(i, i) / (RowWise + ColWise);
            F1 += F1temp;
            actualClasses++;
        }
    }

    F1 = F1 / actualClasses;
    return 100 * F1;
}

/**
 * @brief Calculate fitness function as naive classification accuracy
 * 
 * 
 * @param predict Eigen matrix of predicted values
 * @param actual Eigen matrix of actual values
 * @param nOutput Number of output classe
 * @param OutputLabel [deprecated] return the outputLabel
 * @param rule Unused. Necessary to be compliant with external fitness functions 
 * @return floatBO Value of the fitness function
 */
floatBO EchoBay::Comparator::Accuracy(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    // This part can be put in a function as well
    MatrixBO label(actual.rows(), actual.cols());

    int argMax;
    for (size_t i = 0; i < (size_t)actual.size(); ++i)
    {
        //Compute Argmax to determine the class
        predict.row(i).maxCoeff(&argMax);
        label(i, 0) = ((floatBO)argMax);
    }

    // Compute Confusion Matrix
    Eigen::MatrixXi ConfusionMat(nOutput, nOutput);
    ConfusionMat.setZero();
    ConfusionMatrix(label, actual, nOutput, ConfusionMat);
    std::cout << "\n" << ConfusionMat << std::endl;

    floatBO Accuracy = ConfusionMat.diagonal().sum() / (double)ConfusionMat.sum();
    return 100 * Accuracy;
}

/**
 * @brief Calculate fitness function as network memory capacity
 * 
 * @param predict Eigen matrix of predicted values
 * @param actual Eigen matrix of actual values
 * @param nOutput Number of output classe
 * @param OutputLabel [deprecated] return the outputLabel
 * @param rule Unused. Necessary to be compliant with external fitness functions
 * @return floatBO Value of the fitness function
 */
floatBO EchoBay::Comparator::MemoryCapacity(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    // Variables
    int i;
    ArrayBO demean_actual(actual.rows());
	ArrayBO demean_predict(predict.rows());

    floatBO numerator;
    floatBO var_correction, variance_actual, variance_predict;
    ArrayBO MM(predict.cols());

    // Calculate NRMSE
    var_correction = predict.rows() / (floatBO)(predict.rows() - 1);
    for (i = 0; i < nOutput; i++)
    {
        // Calculate difference array
        demean_actual = (actual.col(i).array() - actual.col(i).array().mean());
		demean_predict = (predict.col(i).array() - predict.col(i).array().mean());
		numerator= (demean_actual * demean_predict).mean();

        variance_actual = var_correction * (actual.col(i).array().pow(2).mean() - pow(actual.col(i).array().mean(), 2));
        variance_predict = var_correction * (predict.col(i).array().pow(2).mean() - pow(predict.col(i).array().mean(), 2));

        MM(i) = pow(numerator,2) /(variance_actual * variance_predict);
    }

    return MM.sum();
}

#if defined(USE_PYBIND)
floatBO EchoBay::Comparator::ExtFitness(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule)
{
    floatBO result = 100;
    try
    {
        // Take GIL if multithreading is running
#ifdef USE_TBB
        gil_mutex.lock();
#endif
        // Import fitness function
        pybind11::object fitnessFunction = pybind11::module::import(rule.c_str()).attr("fitness");

        // Get result
        pybind11::object result_py = fitnessFunction(predict, actual);
        result = result_py.cast<floatBO>();

        // Release the GIL
#ifdef USE_TBB
        gil_mutex.unlock();
#endif
    }
    catch(std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }
    // Return fitness
    return 100 - result;
}
#endif