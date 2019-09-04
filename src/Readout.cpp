#include "Readout.hpp"

/**
 * @brief Calculate Wout using Tikhonow regularization
 * 
 * @param rows Rows of Wout
 * @param cols Columns of Wout
 * @param lambda Regression lambda factor
 * @param biasedState Collection of Reservoir states with added bias
 * @param target Target matrix to be used in regression solution
 * @return MatrixBO Learned Wout
 */
MatrixBO EchoBay::Wout_ridge(int rows, int cols, double lambda, Eigen::Ref<MatrixBO> biasedState, Eigen::Ref<MatrixBO> target)
{
    // Wout initialization
    MatrixBO Wout(rows, cols); //outNr, outCols*nClasses

    // Wout computation
    // pInvState = biasT*bias+I*lambda
    MatrixBO pinvState = biasedState.transpose() * biasedState + MatrixBO::Identity(rows, rows) * lambda;
    pinvState = pinvState.completeOrthogonalDecomposition().pseudoInverse();
    Wout = pinvState.transpose() * biasedState.transpose() * target;

    return Wout;
}

/**
 * @brief Train readout layer based on training data, a given Reservoir and target
 * 
 * @param ESN Reservoir used to process data
 * @param trainData Eigen Matrix training data
 * @param sampleState Eigen Array defining data sampling
 * @param target Target data
 * @param lambda Regression lambda factor
 * @param blockStep Size of data blocks in samples to be used in training. Reduces memory footprint of application and optimize speed.
 * @return MatrixBO Learned Wout
 */
MatrixBO EchoBay::readout_train(Reservoir &ESN, const MatrixBO &trainData,
                                const Eigen::Ref<const ArrayBO> sampleState,
                                Eigen::Ref<MatrixBO> target, double lambda, int blockStep)
{
    // General Configuration
    int nLayers = ESN.get_nLayers();
    int trainSamples = trainData.rows();
    int Nu = trainData.cols();

    //Reservoir Type
    int type = ESN.get_ReservoirType();

    // Get outNr
    int outNr;
    //int lastNr = ESN.get_LayerConfig()[nLayers - 1].Nr + 1;
    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();
    std::vector<ArrayI> NOutindex = ESN.get_WoutIndex();
    int fullNr, fullNrSWT;
    fullNr = std::accumulate(layerConfig.begin(), layerConfig.end(), 0, sumNr) + 1;
    if(type ==1){
        fullNrSWT = std::accumulate(NOutindex.begin(), NOutindex.end(), 0, sumNrSWT) + 1;
        outNr = fullNrSWT;
    }else{
        outNr = fullNr;
    }
    // Add Input dimension
    outNr += Nu;
    
    // Get outDimension
    int outDimension = target.cols();

    // Block size management
    int start = 0, startTarget = 0, blockSize;
    blockSize = trainSamples > blockStep ? blockStep : trainSamples;
    int iterations = (int)ceil(trainSamples / (floatBO)blockSize);

    // Compute State Structures
    MatrixBO trainState;
    ArrayBO u = ArrayBO::Constant(Nu + 1, 1.0);

    // placeHolder Structures
    MatrixBO placeHolderInput = MatrixBO::Zero(blockSize, Nu);
    MatrixBO placeHolderTarget;
    ArrayBO placeHolderSample(blockSize);
    int placeHolderTrainPoints;

    // Wout Structures
    MatrixBO pinvState = MatrixBO::Zero(outNr, outNr);
    MatrixBO stateTarget = MatrixBO::Zero(outNr, outDimension);

    // LOB = Lower Order Bits for Kahan Sum
    MatrixBO pinvLOB = MatrixBO::Zero(outNr, outNr);
    MatrixBO stateTargetLOB = MatrixBO::Zero(outNr, outDimension);

    // Wout structure
    MatrixBO Wout(outNr, outDimension);

    for (int i = 0; i < iterations; ++i)
    {
        // Isolate Input and Sample Vectors
        placeHolderInput = trainData.block(start, 0, blockSize, Nu);
        placeHolderSample = sampleState.segment(start, blockSize);

        // How many Labels to isolate, taking in account how many state will be sampled
        placeHolderTrainPoints = placeHolderSample.count();
        // placeHolderTarget.resize(placeHolderTrainPoints, outDimension); Not needed
        placeHolderTarget = target.block(startTarget, 0, placeHolderTrainPoints, outDimension);

        // How many state will be sampled
        trainState.resize(placeHolderTrainPoints, fullNr);
        // Compute State
        if (nLayers > 1)
        {
            compute_state(trainState, ESN.WinLayers, ESN.WrLayers, placeHolderInput, ESN.stateMat, u, placeHolderSample, layerConfig);
        }
        else
        {
            compute_state(trainState, ESN.WinLayers[0], ESN.WrLayers[0], placeHolderInput, ESN.stateMat[0], u, placeHolderSample, layerConfig[0].leaky);
        }

        MatrixBO reducedState(placeHolderTrainPoints, outNr);
        MatrixBO cleanInput = clean_input(placeHolderInput, placeHolderSample);
        if (type == 1)
        {
            int tempNR = 0;
            int tempSWT = 0;
            for (int i = 0; i < nLayers; ++i)
            {
                for (int j = 0; j < NOutindex[i].size(); ++j)
                {
                    reducedState.col(tempSWT+ j) = trainState.col(tempNR + NOutindex[i](j));
                }
                tempSWT += NOutindex[i].size();
                tempNR  += layerConfig[i].Nr;
            }
            reducedState.block(0, outNr-Nu-1, reducedState.rows(), 1) = trainState.col(fullNr-1);
            reducedState.block(0, outNr-Nu, reducedState.rows(), Nu) = cleanInput;
        }
        else
        {
            reducedState << trainState, cleanInput;
        }
        // "Secure" sum
        kahan_sum(reducedState.transpose() * reducedState, pinvState, pinvLOB);
        kahan_sum(reducedState.transpose() * placeHolderTarget, stateTarget, stateTargetLOB);

        // Basic sum
        //stateTarget = stateTarget + trainState.transpose() * placeHolderTarget;
        //pinvState = pinvState + trainState.transpose() * trainState;

        // Reset and update Starting Points and/or block dimension
        trainState.setZero();
        start += blockSize;
        startTarget += placeHolderTrainPoints;
        blockSize = blockStep < (trainSamples - start) ? blockStep : trainSamples - start;
    }
    // Compute Final Wout
    pinvState = pinvState + MatrixBO::Identity(outNr, outNr) * lambda;
    pinvState = pinvState.completeOrthogonalDecomposition().pseudoInverse();

    Wout = pinvState * stateTarget;

    return Wout;
}

/**
 * @brief Perform prediction using a learned readout
 * 
 * @param ESN Reservoir used to process data
 * @param inputData Input data processed by the ESN
 * @param sampleState Eigen Array defining data sampling
 * @param Wout Readout matrix
 * @param blockStep Size of data blocks in samples to be used in training. Reduces memory footprint of application and optimize speed.
 * @return MatrixBO Prediction results
 */
MatrixBO EchoBay::readout_predict(Reservoir &ESN, const MatrixBO &inputData, 
                                  const Eigen::Ref<const ArrayBO> sampleState,
                                  const Eigen::Ref<MatrixBO> Wout, int blockStep)
{
    // General Configuration
    int nLayers = ESN.get_nLayers();
    int valSamples = inputData.rows();
    int Nu = inputData.cols();

    int type = ESN.get_ReservoirType();

    // get outNr
    int outNr;
    //int lastNr = ESN.get_LayerConfig()[nLayers - 1].Nr + 1;
    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();
    std::vector<ArrayI> NOutindex = ESN.get_WoutIndex();
    int fullNr, fullNrSWT = 0;
    fullNr = std::accumulate(layerConfig.begin(), layerConfig.end(), 0, sumNr) + 1;
    if(type ==1){
        fullNrSWT = std::accumulate(NOutindex.begin(), NOutindex.end(), 0, sumNrSWT) + 1;
        outNr = fullNrSWT;
    }else{
        outNr = fullNr;
    }
    //outNr = fullNr;
    // Add Input Dimension
    outNr += Nu;

    // Get outDimension
    int outDimension = Wout.cols();

    // Block size management
    int start = 0, startTarget = 0, blockSize;
    blockSize = valSamples > blockStep ? blockStep : valSamples;
    int iterations = (int)ceil(valSamples / (floatBO)blockSize);

    // Compute State Structures
    MatrixBO valState;
    ArrayBO u = ArrayBO::Constant(Nu + 1, 1.0);
    //std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();

    // placeHolder Structures
    MatrixBO placeHolderInput = MatrixBO::Zero(blockSize, Nu);
    MatrixBO placeHolderTarget;
    ArrayBO placeHolderSample(blockSize);
    int placeHolderTrainPoints;

    //Prediction Structure
    MatrixBO prediction(sampleState.count(), outDimension);

    for (int i = 0; i < iterations; ++i)
    {
        // Isolate Input and Sample Vectors
        placeHolderInput = inputData.block(start, 0, blockSize, Nu);
        placeHolderSample = sampleState.segment(start, blockSize);
        // How many Labels to isolate, taking in account how many state will be sampled
        placeHolderTrainPoints = placeHolderSample.count();

        // How many state will be sampled
        valState.resize(placeHolderTrainPoints, fullNr);

        // Compute State
        if (nLayers > 1)
        {
            compute_state(valState, ESN.WinLayers, ESN.WrLayers, placeHolderInput, ESN.stateMat, u, placeHolderSample, layerConfig);
        }
        else
        {
            compute_state(valState, ESN.WinLayers[0], ESN.WrLayers[0], placeHolderInput, ESN.stateMat[0], u, placeHolderSample, layerConfig[0].leaky);
        }

        MatrixBO reducedState(placeHolderTrainPoints, outNr);
        MatrixBO cleanInput = clean_input(placeHolderInput, placeHolderSample);
        if (type == 1)
        {
            int tempNR = 0;
            int tempSWT = 0;
            for (int i = 0; i < nLayers; ++i)
            {
                for (int j = 0; j < NOutindex[i].size(); ++j)
                {
                    reducedState.col(tempSWT+ j) = valState.col(tempNR + NOutindex[i](j));

                }
                tempSWT += NOutindex[i].size();
                tempNR  += layerConfig[i].Nr;
            }
            reducedState.block(0, outNr - Nu - 1, reducedState.rows(), 1) = valState.col(fullNr-1);
            reducedState.block(0, outNr - Nu, reducedState.rows(), Nu) = cleanInput;
        }
        else
        {
            reducedState << valState, cleanInput;
        }

        //Compute Prediction
        for (int j = 0; j < placeHolderTrainPoints; ++j)
        {
            prediction.row(startTarget + j) = reducedState.row(j).matrix() * Wout;
        }

        // Reset and update Starting Points and/or block dimension
        valState.setZero();
        start += blockSize;
        startTarget += placeHolderTrainPoints;
        blockSize = blockStep < (valSamples - start) ? blockStep : valSamples - start;
    }

    return prediction;
}

/**
 * @brief Perform single prediction
 * 
 * @param Wout Learned readout
 * @param ESNState Single ESN state
 * @return ArrayBO Prediction
 */
ArrayBO EchoBay::readout_predict(const Eigen::Ref<MatrixBO> Wout, const Eigen::Ref<const ArrayBO> ESNState)
{
    return ESNState.matrix() * Wout;
}

/**
 * @brief Secure Kahan summation to be used in training states update
 * 
 * @param Input New Data
 * @param Sum Current Sum
 * @param C Lower Order Bits Matrix
 */
void EchoBay::kahan_sum(MatrixBO Input, Eigen::Ref<MatrixBO> Sum, Eigen::Ref<MatrixBO> C)
{
    MatrixBO y = Input - C;
    MatrixBO t = Sum + y;
    C = (t - Sum) - y;
    Sum = t;
}

/**
 * @brief Select input based on sampleState array
 * 
 * @param input Input matrix
 * @param sampleState Sampling array
 * @return MatrixBO Cleaned data
 */
MatrixBO EchoBay::clean_input(const Eigen::Ref<const MatrixBO> input, const Eigen::Ref<const ArrayBO> sampleState)
{
    int outRows = 0;
    MatrixBO inputCleaned(sampleState.count(), input.cols());
        for (int s = 0; s < sampleState.rows(); s++)
        {
            if (sampleState(s) != 0)
            {
                inputCleaned.row(outRows) = input.row(s);
                outRows++;
            }
        }
        
    return inputCleaned;
}