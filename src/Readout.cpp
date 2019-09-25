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
 * @param store DataStorage containing input and sampling data
 * @param target Target data
 * @param lambda Regression lambda factor
 * @param blockStep Size of data blocks in samples to be used in training. Reduces memory footprint of application and optimize speed.
 * @return MatrixBO Learned Wout
 */
MatrixBO EchoBay::readout_train(Reservoir &ESN, const DataStorage &store,
                                Eigen::Ref<MatrixBO> target, double lambda,
                                int blockStep)
{
    // General Configuration
    int nLayers = ESN.get_nLayers();
    int trainSamples = store.get_dataLength(EchoBay::Train);
    int Nu = store.get_dataCols(EchoBay::Train);

    // Get trainData
    MatrixBO trainData = store.get_data(EchoBay::Train, EchoBay::selData);

    //Reservoir Type
    int type = ESN.get_ReservoirType();

    // Get outNr
    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();
    std::vector<ArrayI> NOutindex = ESN.get_WoutIndex();
    int fullNr = ESN.get_fullNr();
    int outNr = (type == 1) ? ESN.get_NrSWT() : fullNr; 
    // Add Input dimension
    outNr += Nu;

    // Get outDimension
    int outDimension = target.cols();

    // Wout Structures
    MatrixBO pinvState = MatrixBO::Zero(outNr, outNr);
    MatrixBO stateTarget = MatrixBO::Zero(outNr, outDimension);

    // LOB = Lower Order Bits for Kahan Sum
    MatrixBO pinvLOB = MatrixBO::Zero(outNr, outNr);
    MatrixBO stateTargetLOB = MatrixBO::Zero(outNr, outDimension);

    // Wout structure
    MatrixBO Wout(outNr, outDimension);

    // Get number of independent sampleBatches
    std::vector<ArrayI8> sampleBatches = store.get_samplingBatches(EchoBay::Train);
    int nBatches = sampleBatches.size();

    // Iterate on nBatches
#pragma omp parallel for
    for (int i = 0; i < nBatches; i++)
    {
        // Block size management
        int batchSamples = sampleBatches[i].rows();
        int start = 0, startSample = 0, startTarget = 0, blockSize;
        blockSize = batchSamples > blockStep ? blockStep : batchSamples;
        int iterations = (int)ceil(batchSamples / (floatBO)blockSize);

        // Compute State Structures
        MatrixBO trainState;

        // placeHolder Structures
        MatrixBO placeHolderInput = MatrixBO::Zero(blockSize, Nu);
        MatrixBO placeHolderTarget;
        ArrayI8 placeHolderSample(blockSize);
        int placeHolderTrainPoints;

        // Identify starting point
        start = store.get_dataOffset(EchoBay::Train, i);
        startTarget = store.get_maxSamples(EchoBay::Train, i);

        // Create placeholder states
        std::vector<ArrayBO> placeHolderState;
        placeHolderState = ESN.stateMat;

        for (int j = 0; j < iterations; j++)
        {
            // Isolate Input and Sample Vectors
            placeHolderInput = trainData.block(start, 0, blockSize, Nu);
            placeHolderSample = sampleBatches[i].segment(startSample, blockSize);

            // How many Labels to isolate, taking in account how many state will be sampled
            placeHolderTrainPoints = placeHolderSample.count();
            // placeHolderTarget.resize(placeHolderTrainPoints, outDimension); Not needed
            placeHolderTarget = target.block(startTarget, 0, placeHolderTrainPoints, outDimension);

            // How many state will be sampled
            trainState.resize(placeHolderTrainPoints, fullNr);
            trainState.setZero();

            // Compute State
            if (nLayers > 1)
            {
                compute_state(trainState, ESN.WinLayers, ESN.WrLayers, placeHolderInput, placeHolderState, placeHolderSample, layerConfig);
            }
            else
            {
                compute_state(trainState, ESN.WinLayers[0], ESN.WrLayers[0], placeHolderInput, placeHolderState[0], placeHolderSample, layerConfig[0].leaky);
            }

            MatrixBO reducedState = MatrixBO::Zero(placeHolderTrainPoints, outNr);
            MatrixBO cleanInput = clean_input(placeHolderInput, placeHolderSample);
            if (type == 1)
            {
                int tempNR = 0;
                int tempSWT = 0;
                for (int k = 0; k < nLayers; ++k)
                {
                    for (int l = 0; l < NOutindex[k].size(); ++l)
                    {
                        reducedState.col(tempSWT + l) = trainState.col(tempNR + NOutindex[k](l));
                    }
                    tempSWT += NOutindex[k].size();
                    tempNR += layerConfig[k].Nr;
                }
                reducedState.block(0, outNr - Nu - 1, reducedState.rows(), 1) = trainState.col(fullNr - 1);
                reducedState.block(0, outNr - Nu, reducedState.rows(), Nu) = cleanInput;
            }
            else
            {
                reducedState << trainState, cleanInput;
            }

#pragma omp critical
            {
                // "Secure" sum
                kahan_sum(reducedState.transpose() * reducedState, pinvState, pinvLOB);
                kahan_sum(reducedState.transpose() * placeHolderTarget, stateTarget, stateTargetLOB);
                // Basic sum
                //stateTarget = stateTarget + trainState.transpose() * placeHolderTarget;
                //pinvState = pinvState + trainState.transpose() * trainState;
            }

            // Reset and update Starting Points and/or block dimension
            start += blockSize;
            startTarget += placeHolderTrainPoints;
            startSample += blockSize;
            blockSize = blockStep < (trainSamples - start) ? blockStep : trainSamples - start;
        }
        //Save state TODO check if it is still necessary
        ESN.stateMat = placeHolderState;
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
 * @param store DataStorage containing input and sampling data
 * @param Wout Readout matrix
 * @param blockStep Size of data blocks in samples to be used in training. Reduces memory footprint of application and optimize speed.
 * @return MatrixBO Prediction results
 */
MatrixBO EchoBay::readout_predict(Reservoir &ESN, const DataStorage &store,
                                  const Eigen::Ref<MatrixBO> Wout, int blockStep)
{
    // General Configuration
    int nLayers = ESN.get_nLayers();
    int valSamples = store.get_dataLength(EchoBay::Valid);
    int Nu = store.get_dataCols(EchoBay::Valid);

    // Get trainData
    MatrixBO evalData = store.get_data(EchoBay::Valid, EchoBay::selData);

    // Get Reservoir type
    int type = ESN.get_ReservoirType();

    // get outNr
    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();
    std::vector<ArrayI> NOutindex = ESN.get_WoutIndex();
    int fullNr = ESN.get_fullNr();
    int outNr = (type == 1) ? ESN.get_NrSWT() : fullNr; 
    // Add Input Dimension
    outNr += Nu;

    // Get outDimension
    int outDimension = Wout.cols();

    // Get number of independent sampleBatches
    std::vector<ArrayI8> sampleBatches = store.get_samplingBatches(EchoBay::Valid);
    int nBatches = sampleBatches.size();

    // Prediction Structure
    int stateCount = store.get_maxSamples(EchoBay::Valid);
    MatrixBO prediction(stateCount, outDimension);

    // Iterate on nBatches
#pragma omp parallel for
    for (int i = 0; i < nBatches; i++)
    {
        // Block size management
        int batchSamples = sampleBatches[i].rows();
        int start = 0, startSample = 0, startTarget = 0, blockSize;
        blockSize = batchSamples > blockStep ? blockStep : batchSamples;
        int iterations = (int)ceil(batchSamples / (floatBO)blockSize);

        // Compute State Structures
        MatrixBO valState;

        // placeHolder Structures
        MatrixBO placeHolderInput = MatrixBO::Zero(blockSize, Nu);
        MatrixBO placeHolderTarget;
        ArrayI8 placeHolderSample(blockSize);
        int placeHolderTrainPoints;

        // Identify starting point
        start = store.get_dataOffset(EchoBay::Valid, i);
        startTarget = store.get_maxSamples(EchoBay::Valid, i);

        // Create placeholder states
        std::vector<ArrayBO> placeHolderState;
        placeHolderState = ESN.stateMat;

        // Iterate the single batch
        for (int j = 0; j < iterations; j++)
        {
            // Isolate Input and Sample Vectors
            placeHolderInput = evalData.block(start, 0, blockSize, Nu);
            placeHolderSample = sampleBatches[i].segment(startSample, blockSize); //sampleState.segment(start, blockSize);

            // How many Labels to isolate, taking in account how many state will be sampled
            placeHolderTrainPoints = placeHolderSample.count();

            // How many state will be sampled
            valState.resize(placeHolderTrainPoints, fullNr);
            valState.setZero();

            // Compute State
            if (nLayers > 1)
            {
                compute_state(valState, ESN.WinLayers, ESN.WrLayers, placeHolderInput, placeHolderState, placeHolderSample, layerConfig);
            }
            else
            {
                compute_state(valState, ESN.WinLayers[0], ESN.WrLayers[0], placeHolderInput, placeHolderState[0], placeHolderSample, layerConfig[0].leaky);
            }

            MatrixBO reducedState = MatrixBO::Zero(placeHolderTrainPoints, outNr);
            MatrixBO cleanInput = clean_input(placeHolderInput, placeHolderSample);
            if (type == 1)
            {
                int tempNR = 0;
                int tempSWT = 0;
                for (int k = 0; k < nLayers; ++k)
                {
                    for (int l = 0; l < NOutindex[k].size(); ++l)
                    {
                        reducedState.col(tempSWT + l) = valState.col(tempNR + NOutindex[k](l));
                    }
                    tempSWT += NOutindex[k].size();
                    tempNR += layerConfig[k].Nr;
                }
                reducedState.block(0, outNr - Nu - 1, reducedState.rows(), 1) = valState.col(fullNr - 1);
                reducedState.block(0, outNr - Nu, reducedState.rows(), Nu) = cleanInput;
            }
            else
            {
                reducedState << valState, cleanInput;
            }

#pragma omp critical
            {
                //Compute Prediction
                for (int k = 0; k < placeHolderTrainPoints; ++k)
                {
                    prediction.row(startTarget + k) = reducedState.row(k).matrix() * Wout;
                }
            }

            // Reset and update Starting Points and/or block dimension
            start += blockSize;
            startTarget += placeHolderTrainPoints;
            blockSize = blockStep < (valSamples - start) ? blockStep : valSamples - start;
        }
        //Save state TODO check if it is still necessary
        ESN.stateMat = placeHolderState;
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
MatrixBO EchoBay::clean_input(const Eigen::Ref<const MatrixBO> input, const Eigen::Ref<const ArrayI8> sampleState)
{
    int outRows = 0;
    MatrixBO inputCleaned = MatrixBO::Zero(sampleState.count(), input.cols());
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