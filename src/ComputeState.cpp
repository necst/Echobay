#include "ComputeState.hpp"
using namespace Eigen;

/**
 * @brief Perform ESN state computation on data using naive update
 * 
 * @param dst Destination Matrix
 * @param Win Input scaling Matrix
 * @param Wr Reservoir's weights
 * @param src Input data
 * @param stateArr ESN state Array
 * @param sampleState Eigen Array that manages state sampling
 */
void EchoBay::compute_state(Eigen::Ref<MatrixBO> dst, Eigen::Ref<MatrixBO> Win,
                           Eigen::Ref<SparseBO> Wr, const MatrixBO &src,
                           Eigen::Ref<ArrayBO> stateArr,
                           const Eigen::Ref<const ArrayI8> sampleState)
{
    size_t trainSamples = src.rows();
    MatrixBO u(src.cols()+1, src.rows());
    u << src.transpose(), MatrixBO::Ones(1, src.rows()); // TODO improve this to avoid transpose
    int sample = 0;
    // Computation
    for (size_t t = 0; t < trainSamples; ++t)
    {
        // Compute new state
        stateArr = Win * u.col(t) + Wr * stateArr.matrix();
        stateArr = stateArr.tanh();

        // Save state for later usage
        if (sampleState(t) != 0)
        {
            dst.row(sample) << stateArr.transpose(), 1.0;
            sample++;
            if (sampleState(t) == -1)
            {
                stateArr.setZero();
            }
        }
    }
}

/**
 * @brief Perform ESN state computation on data using leaky update
 * 
 * @param dst Destination Matrix
 * @param Win Input scaling Matrix
 * @param Wr Reservoir's weights
 * @param src Input data
 * @param stateArr ESN state Array
 * @param sampleState Eigen Array that manages state sampling
 * @param leaky Leaky integration factor
 */
void EchoBay::compute_state(Eigen::Ref<MatrixBO> dst, Eigen::Ref<MatrixBO> Win,
                           Eigen::Ref<SparseBO> Wr, const MatrixBO &src,
                           Eigen::Ref<ArrayBO> stateArr,
                           const Eigen::Ref<const ArrayI8> sampleState, floatBO leaky)
{
    size_t trainSamples = src.rows();
    MatrixBO u(src.cols()+1, src.rows());
    u << src.transpose(), MatrixBO::Ones(1, src.rows()); // TODO improve this to avoid transpose
    int sample = 0;
    // Leaky state
    ArrayBO prevState(stateArr.rows());
    // Leaky factor
    floatBO leakyFactor = (1.0 - leaky);
    // Computation
    for (size_t t = 0; t < trainSamples; ++t)
    {
        // Update leaky state
        prevState = leakyFactor * stateArr;

        // Compute new state
        stateArr = Win * u.col(t) + Wr * stateArr.matrix();
        stateArr = prevState + leaky * stateArr.tanh();

        // Save state for later usage
        if (sampleState(t) != 0)
        {
            dst.row(sample) << stateArr.transpose(), 1.0;
            sample++;
            if (sampleState(t) == -1)
            {
                stateArr.setZero();
            }
        }
    }
}

/**
 * @brief Perform ESN state computation on data using leaky update and deep layers
 * 
 * @param dst Destination Matrix
 * @param WinL Vector containing input scaling Matrices
 * @param WrL Vector containing reservoir weights matrices
 * @param src Input data
 * @param stateMat ESN state Matrix, one Array for each layer
 * @param sampleState Eigen Array that manages state sampling
 * @param layerConfig Layers configuration vector
 */
void EchoBay::compute_state(Eigen::Ref<MatrixBO> dst, const std::vector<MatrixBO> &WinL,
                           const std::vector<SparseBO> &WrL,
                           const MatrixBO &src, std::vector<ArrayBO> &stateMat,
                           const Eigen::Ref<const ArrayI8> sampleState,
                           const std::vector<layerParameter> &layerConfig)
{
    // Variables
    size_t trainSamples = src.rows();
    MatrixBO u(src.cols()+1, src.rows());
    u << src.transpose(), MatrixBO::Ones(1, src.rows()); // TODO improve this to avoid transpose
    int nLayers = WinL.size();
    int sample = 0;
    int i;

    // Leaky state
    // Allocate deep states
    std::vector<ArrayBO> prevState;
    prevState.reserve(nLayers);
    std::vector<ArrayBO> layerInput;
    layerInput.reserve(nLayers - 1);
    prevState.push_back(stateMat[0]);
    for (i = 1; i < nLayers; i++)
    {
        prevState.push_back(stateMat[i]);
        ArrayBO inputSupport(stateMat[i - 1].rows() + 1);
        inputSupport << stateMat[i - 1], 1.0;
        layerInput.push_back(inputSupport);
    }

    // Leaky factor
    ArrayBO leakyFactor(nLayers);
    ArrayBO leaky(nLayers);
    i = 0;
    for(const auto& layer: layerConfig)
    {
        leaky[i] = layer.leaky;
        ++i;
    }
    leakyFactor = 1.0 - leaky;

    // Computation
    for (size_t t = 0; t < trainSamples; ++t)
    {
        // Update leaky state
        prevState[0] = leakyFactor[0] * stateMat[0];

        // Compute new state
        stateMat[0] = WinL[0] * u.col(t) + WrL[0] * stateMat[0].matrix();
        stateMat[0] = prevState[0] + leaky[0] * stateMat[0].tanh();
        for (i = 1; i < nLayers; ++i)
        {
            // Update leaky state
            prevState[i] = leakyFactor[i] * stateMat[i];
            layerInput[i - 1] << stateMat[i - 1], 1.0;

            // Compute new state
            stateMat[i] = WinL[i] * layerInput[i - 1].matrix() + WrL[i] * stateMat[i].matrix();
            stateMat[i] = prevState[i] + leaky[i] * stateMat[i].tanh();
        }
        // Save state for later usage
        if (sampleState(t) != 0)
        {
            int j = 0;
            for(const auto& state: stateMat)
            {
                dst.row(sample).segment(j, state.size()) = state.transpose();
                j += state.size();
            }
            dst.row(sample).tail(1) << 1.0;
            sample++;
            if (sampleState(t) == -1)
            {
                for (auto& state : stateMat)
                {
                    state.setZero();
                }
            }
        }
    }
}

/**
 * @brief Single state update with leaky integration and deep layers
 * 
 * @param WinL Vector containing input scaling Matrices
 * @param WrL Vector containing reservoir weights matrices
 * @param src Input data
 * @param stateMat ESN state Matrix, one Array for each layer
 * @param layerConfig Layers configuration vector
 * @return ArrayBO Output state of the last layer
 */
ArrayBO EchoBay::update_state(const std::vector<MatrixBO> &WinL,
                              const std::vector<SparseBO> &WrL,
                              const Eigen::Ref<ArrayBO> src, std::vector<ArrayBO> &stateMat,
                              const std::vector<layerParameter> &layerConfig)
{
    // Variables
    int nLayers = WinL.size();
    int i;

    // Leaky state
    // Allocate deep states
    std::vector<ArrayBO> prevState;
    prevState.reserve(nLayers);
    std::vector<ArrayBO> layerInput;
    layerInput.reserve(nLayers - 1); //TODO check if we need to save this
    prevState.push_back(stateMat[0]);
    for (i = 1; i < nLayers; i++)
    {
        prevState.push_back(stateMat[i]);
        ArrayBO inputSupport(stateMat[i - 1].rows() + 1);
        inputSupport << stateMat[i - 1], 1.0;
        layerInput.push_back(inputSupport);
    }

    // Leaky factor
    ArrayBO leakyFactor(nLayers);
    ArrayBO leaky(nLayers);
    i = 0;
    for(const auto& layer: layerConfig)
    {
        leaky[i] = layer.leaky;
        ++i;
    }
    leakyFactor = 1.0 - leaky;

    // update input
    ArrayBO u = ArrayBO::Constant(src.cols() + 1, 1.0);
    size_t uSize = src.cols();
    u.head(uSize) = src;

    // Update leaky state
    prevState[0] = leakyFactor[0] * stateMat[0];

    // Compute new state
    stateMat[0] = WinL[0] * u.matrix() + WrL[0] * stateMat[0].matrix();
    stateMat[0] = prevState[0] + leaky[0] * stateMat[0].tanh();
    for (i = 1; i < nLayers; ++i)
    {
        // Update leaky state
        prevState[i] = leakyFactor[i] * stateMat[i];
        layerInput[i - 1] << stateMat[i - 1], 1.0;

        // Compute new state
        stateMat[i] = WinL[i] * layerInput[i - 1].matrix() + WrL[i] * stateMat[i].matrix();
        stateMat[i] = prevState[i] + leaky[i] * stateMat[i].tanh();
    }

    // Return final state TODO decide how to manage this in DeepESN
    return stateMat[nLayers - 1];
}

/**
 * @brief  Compute the ESN output. Currently deprecated by Readout methods.
 * 
 * @param Win 
 * @param Wr 
 * @param Wout 
 * @param input_data 
 * @param start_state 
 * @param Nu 
 * @param n_outputs 
 */
void EchoBay::compute_output(const Eigen::Ref<MatrixBO> &Win, const Eigen::Ref<Eigen::SparseMatrix<floatBO, RowMajor>> &Wr,
                             const Eigen::Ref<MatrixBO> &Wout, const std::vector<ArrayBO> input_data,
                             const Eigen::Ref<ArrayBO> start_state, const unsigned int Nu, const unsigned int n_outputs)
{
    /*// Input temporary variable
    Array2f u(Nu);
    u(1) = 1.0;

    // Initialize state
    ArrayXf current_state;
    current_state.resize(start_state.rows());
    ArrayXf biased_state;
    current_state << start_state;
    biased_state.resize(start_state.rows() + 1);

    // Initialize output
    MatrixBO output_value(n_outputs, 1);

    int compute_samples = input_data.size();
    for (int t = 0; t <= compute_samples; t++)
    {
        // update input
        u(0) = input_data[t];

        // Compute new state
        current_state = Win * u.matrix() + Wr * current_state.matrix();
        current_state = current_state.tanh();

        // Compute output
        biased_state << current_state, 1.0;
        output_value = Wout.transpose() * biased_state.matrix();
    }*/
}