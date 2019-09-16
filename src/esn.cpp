#include "esn.hpp"
#include <chrono>

using namespace Eigen;
using namespace EchoBay;
using namespace std::chrono;

/**
 * @brief Main optimization function. Configure or load an Echo State Network
 * 
 * @param optParams Eigen Vector used by Limbo to define hyper-parameters
 * @param confParams YAML Node containing hyper-parameters at high level
 * @param outputFolder Path to save all output files
 * @param store DataStorage object containing training data and labels. See also EchoBay::DataStorage
 * @param guessEval If true evaluates multiple guesses as the worst case, otherwise, get the mean of guesses
 * @param computationType "train" performs ESN training, otherwise, validation and testing
 * @param matrixFolder Path to the folder where Reservoir matrices will be saved or loaded
 * @return ArrayBO Return fitness value to the Bayesian Optimizer
 */
ArrayBO EchoBay::esn_caller(const Eigen::VectorXd &optParams, YAML::Node confParams, 
                       std::string outputFolder, const DataStorage &store, 
                       bool guessEval, const std::string &computationType, 
                       const std::string &matrixFolder)
{
    // Number of guesses
    int guesses = confParams["Guesses"].as<int>();

    MatrixBO fitnessMatrix(guesses, 2);
    ArrayBO fitnessOut;

    // Problem settings
    double lambda = parse_config<double>("lambda", -1, 0, optParams, confParams, 0);
    std::string problemType = confParams["Problem_Definition"]["type"].as<std::string>();
    std::string fitnessRule = confParams["Problem_Definition"]["Fitness_Function"].as<std::string>();
    // Readout blockStep
    int blockStep = confParams["blockStep"].as<int>();
    // Limits
    int Nu = store._trainData.cols() + 1;

    // Build Reservoir object
    if (computationType == "train")
    {
#ifdef _OPENMP
    omp_lock_t guessLock;
    omp_init_lock(&guessLock);
#pragma omp parallel for
#endif

        for (int i = 0; i < guesses; i++)
        {
            Reservoir ESN = esn_config(optParams, confParams, Nu, i, "");
            // Call rest of execution
            fitnessMatrix.row(i) = esn_train(confParams, ESN, lambda, problemType, fitnessRule, outputFolder, store, guessEval, i);
#ifdef _OPENMP
            omp_set_lock(&guessLock);
#endif
            std::cout << "Guess: " << i << " ";
            std::cout << "multifitness " << fitnessMatrix.row(i).transpose() << std::endl;

#ifdef _OPENMP
            omp_unset_lock(&guessLock);
#endif
        }

        // Calculate output: Train = Mean, Test = Worst Case
        if (guessEval)
        {
            fitnessOut = fitnessMatrix.colwise().minCoeff();
        }
        else
        {
            fitnessOut = fitnessMatrix.colwise().mean();
        }
    }
    else
    {
        std::cout << "Testing optimal configuration on unknown data" << std::endl;
        for (int i = 0; i < guesses; i++)
        {
            Reservoir ESN = esn_config(optParams, confParams, Nu, i, matrixFolder);
            std::string tempName = matrixFolder + "/Wout_eigen.dat";
            MatrixBO Wout;
            read_matrix(tempName, Wout);
            fitnessMatrix.row(i) = esn_compute(confParams, ESN, store, problemType, fitnessRule, blockStep, Wout, outputFolder, guessEval, i);
        }

        // Only worst case in compute
        fitnessOut = fitnessMatrix.colwise().minCoeff();
    }

    std::cout << "Final multiFitness :" << fitnessOut.transpose() << std::endl;
    return fitnessOut;
}

/**
 * @brief Configure or load an Echo State Network
 * 
 * @param optParams Eigen Vector used by Limbo to define hyper-parameters
 * @param confParams YAML Node containing hyper-parameters at high level
 * @param Nu Number of input channels
 * @param guess Number of guess (used for output)
 * @param folder Path to the folder where Reservoir matrices will be saved or loaded
 * @return Reservoir Configured Reservoir object
 */
Reservoir EchoBay::esn_config(const Eigen::VectorXd &optParams, YAML::Node confParams, int Nu, int guess, const std::string folder)
{
    // ESN general Parameters
    // Deep ESN
    int nLayers = parse_config<int>("Nl", -1, 0, optParams, confParams, 1);
    // Esn Topology
    int type = 0;
    if (confParams["ESNType"])
    {
        type = confParams["ESNType"].as<int>();
    }
    // Sampling
    int nWashout = parse_config<int>("washout_sample", -1, 0, optParams, confParams, 10);
    // Problem settings
    double lambda = parse_config<double>("lambda", -1, 0, optParams, confParams, 0);

    // Build Reservoir object
    Reservoir ESN(nLayers, Nu, type);

    // Configure ESN
    ESN.init_LayerConfig(optParams, confParams);

    if (folder != "")
    {
        // Initialize Echo State Network: Win, Wr, stateMat
        ESN.load_network(folder);
    }
    else
    {
        ESN.init_network();
    }

    // Print parameters
    if (guess == 0)
    {
        std::cout << "\nTopology: "<< ESNTypes[type] << std::endl;
        ESN.print_params(nLayers, nWashout, lambda);
    }  

    return ESN;
}

/**
 * @brief Compute the output of a loaded Echo State Network
 * 
 * @param confParams YAML Node containing hyper-parameters at high level
 * @param ESN Reservoir Object
 * @param store Datastorage container with the data to be processed
 * @param problemType Classification, Regression or MemoryCapacity
 * @param fitnessRule Fitness function dependent from the chosen problemType
 * @param blockStep Dimension of the blocks used in Readout prediction
 * @param Wout Readout matrix
 * @param outputFolder Path to the folder where the output will be saved
 * @param saveflag If true, save the output on a file
 * @param guesses Number of guesses to be evaluated (used for output naming)
 * @return ArrayBO Return fitness value to the Bayesian Optimizer
 */
ArrayBO EchoBay::esn_compute(YAML::Node confParams, Reservoir &ESN, const DataStorage &store, 
                             const std::string &problemType, const std::string &fitnessRule, 
                             int blockStep, Eigen::Ref<MatrixBO> Wout,
                             const std::string &outputFolder, bool saveflag, int guesses)
{

    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();
    Comparator fitter(problemType, fitnessRule);
    fitter.set_targetLabel(store._evalLabel, store._evalSampling);

    int Nu = store._evalData.cols() + 1; // Extract info on number of columns of train data + bias vector

    ArrayBO u = ArrayBO::Constant(Nu, 1.0);

    // Compute Prediction
    MatrixBO prediction = readout_predict(ESN, store._evalData, store._evalSampling, Wout, blockStep);

    // Calculate fitness
    // Resize output
    fitter.set_label_size(prediction.rows(), prediction.cols());
    // Get fitness
    floatBO fitness = fitter.get_fitness(prediction);
    MatrixBO outputLabel;

    outputLabel = fitter.get_outputlabel();

    // Consider memory penalization
    floatBO memory = ESN.return_net_dimension(confParams);
    ArrayBO multiFitness(2);

    multiFitness(0) = fitness;
    multiFitness(1) = (memory);
    std::cout << "Computation ended \n";

    if (saveflag)
    {
        std::string nameOut;
        nameOut = "/outputLabel_" + std::to_string(guesses) + ".csv";
        write_matrix<MatrixBO>(outputFolder + std::string(nameOut), prediction);

        // TODO move this to compute prediction
        //nameOut = "/final_state_" + std::to_string(guesses) + ".dat";
        //write_array<ArrayBO>(outputFolder + std::string(nameOut), valState.row(valPoints - 1));
    }
    //return fitness - MemoryPenFactor * Memory;
    return multiFitness;
}

/**
 * @brief Perform Echo State Network training
 * 
 * @param confParams YAML Node containing hyper-parameters at high level
 * @param ESN Reservoir Object
 * @param lambda Regression lambda factor see EchoBay::readout_train(Reservoir &ESN, const MatrixBO &trainData,
                                const Eigen::Ref<const ArrayBO> sampleState,
                                Eigen::Ref<MatrixBO> target, double lambda, int blockStep)
 * @param problemType Classification, Regression or MemoryCapacity
 * @param fitnessRule Fitness function dependent from the chosen problemType
 * @param outputFolder Path to the folder where the output will be saved
 * @param store DataStorage object containing training data and labels. See also EchoBay::DataStorage
 * @param saveflag If true, save the output and trained ESN matrices on a file
 * @param guesses Number of guesses to be evaluated (used for output naming)
 * @return ArrayBO Return fitness value to the Bayesian Optimizer
 */
ArrayBO EchoBay::esn_train(YAML::Node confParams, Reservoir &ESN, double lambda,
                      std::string problemType, std::string fitnessRule,
                      std::string outputFolder, const DataStorage &store, bool saveflag, 
                      int guesses)
{
    // Data size
    int Nu = store._trainData.cols() + 1; // Extract info on number of columns of train data + bias vector

    // ESN general Parameters
    // Deep ESN
    int nLayers = ESN.get_nLayers();
    // Readout blockStep
    int blockStep = confParams["blockStep"].as<int>();

    // Get the configuration of each layer
    std::vector<layerParameter> layerConfig = ESN.get_LayerConfig();

    // Initialize input array with bias
    ArrayBO u = ArrayBO::Constant(Nu, 1.0);

    // Construct comparator class
    Comparator fitter(problemType, fitnessRule);

    // Calculate readout matrix depending on the problem
    MatrixBO targetMatrix = fitter.get_targetMatrix(store._trainLabel, store._trainSampling);

    // Calculate Wout as readout layer
    MatrixBO Wout = readout_train(ESN, store._trainData, store._trainSampling, targetMatrix, lambda, blockStep);
    // Calculate prediction states
    // stateMat is already initialized with last training state

    // If saveflag is set, save the trainingState before being overwritten by evalState
    std::vector<ArrayBO> saveState;
    if (saveflag)
    {
        saveState.reserve(nLayers);
        saveState = ESN.stateMat;
    }

    MatrixBO prediction = readout_predict(ESN, store._evalData, store._evalSampling, Wout, blockStep);

    // Calculate fitness
    // Resize output
    fitter.set_label_size(prediction.rows(), prediction.cols());
    // Get fitness
    fitter.set_targetLabel(store._evalLabel, store._evalSampling);
    floatBO fitness = fitter.get_fitness(prediction);

    MatrixBO outputLabel;
    outputLabel = fitter.get_outputlabel();

    // Consider memory penalization
    floatBO memory = ESN.return_net_dimension(confParams);
    ArrayBO multiFitness(2);
    multiFitness(0) = fitness;
    multiFitness(1) = (memory);
    //std::cout << "multifitness " << multiFitness.transpose() << std::endl;

    // Flag used to save just when the parameters have been chosen
    // See main.cpp for clarity
    if (saveflag)
    {
        if (!boost::filesystem::exists(outputFolder))
        {
            boost::filesystem::create_directories(outputFolder);
        }
        //write_matrix<MatrixBO>(outputFolder + std::string("/State.csv"), biasedState);
        std::string tempName;
        for (int i = 0; i < nLayers; ++i)
        {
            tempName = "/Win_eigen" + std::to_string(i) + ".mtx";
            //saveMarket(ESN.WinLayers[0], outputFolder + tempName);
            write_matrix<MatrixBO>(outputFolder + tempName, ESN.WinLayers[i]);
            tempName = "/Wr_eigen" + std::to_string(i) + ".mtx";
            //write_matrix<MatrixBO>(outputFolder + tempName, ESN.WrLayers[i]);
            saveMarket(ESN.WrLayers[i], outputFolder + tempName);

            tempName = "/State_eigen" + std::to_string(i) + ".mtx";
            write_matrix<MatrixBO>(outputFolder + tempName, saveState[i]);
            //saveMarket(Wout, outputFolder + std::string("/Wout_eigen.mtx"));
        }

        std::string nameOut;
        nameOut = "/outputLabel_" + std::to_string(guesses) + ".csv";
        write_matrix<MatrixBO>(outputFolder + std::string(nameOut), prediction);

        nameOut = "/final_state_" + std::to_string(guesses) + ".dat";
        write_array<ArrayBO>(outputFolder + std::string("/final_state.dat"), saveState[nLayers - 1]);

        write_matrix<MatrixBO>(outputFolder + std::string("/Wout_eigen.dat"), Wout);

        if(confParams["Guesses"].as<int>() == (guesses+1))
        {
            std::cout << "Computation ended \n";
        }
        
    }

    //return fitness - MemoryPenFactor * Memory;
    return multiFitness;
}
