#include "esn.hpp"
#include "IOUtils.hpp"
#include "LimboParams.hpp"
#if defined(USE_PYBIND)
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#endif

using namespace std::chrono;
using namespace limbo;
using namespace EchoBay;

// Limbo evaluation function
struct Eval
{
    // Static parameters
    // Size of output parameters: fitness
    BO_PARAM(size_t, dim_out, 2);

    // Dynamic parameters
    BO_DYN_PARAM(int, dim_in);
    BO_DYN_PARAM(int, data_it);

    // Optimized function
    Eigen::VectorXd operator()(const Eigen::VectorXd &x) const
    {
        // Set sampling with respect to washout
        std::string problemType = configData["Problem_Definition"]["type"].as<std::string>();

        int nWashout = parse_config<int>("washout_sample", -1, 0 , x, configData, 10);
        series.set_sampleArray(samplingTrain, nWashout, true, problemType, "train");
        bool init_flag = samplingTrain(samplingTrain.rows()-1, 1) == 0;
        series.set_sampleArray(samplingVal, nWashout, init_flag, problemType, "valid");
        
        // Memory optimization penalty
        double penalty = 0;
        if (configData["Memory_Optimization_Coeff"]){
        	penalty = configData["Memory_Optimization_Coeff"].as<double>();
        }

        // Call evaluation function
        Eigen::Vector2d Y_vector;
        ArrayBO y = esn_caller(x, configData, ".", series, false,  "train", "");
        time_stats.tSamples.push_back(high_resolution_clock::now());        
        
        Y_vector(0) = y(0) * (1-penalty) + y(1) * penalty * -100;
        Y_vector(1) = 0; //* Y_vector(0); // Penalize up to (penalty)% of the actual fitness
 
        return Y_vector;
    }
};

// Limbo dynamic parameters
BO_DECLARE_DYN_PARAM(int, Eval, dim_in);
BO_DECLARE_DYN_PARAM(int, Eval, data_it);
BO_DECLARE_DYN_PARAM(int, Params::init_randomsampling, samples);
BO_DECLARE_DYN_PARAM(float, Params::stop_maxfitness, threshold);

int set_optDimension()
{
    // Initialize optDimension
    int optDimension = configData["input_dimension_layer"].as<int>();

    // Add nLayers dimension
    int nLayers = 1;
    if (configData["Nl"])
    {   
        nLayers = configData["Nl"]["value"].as<int>();
        optDimension = optDimension * nLayers;
    }

    // Add input dimension general
    optDimension += configData["input_dimension_general"].as<int>();

    // Optimize scaleIn separately
    std::string scaleInType = configData["scaleIn"]["type"].as<std::string>();
    if (scaleInType == "dynamic")
    {
        if(configData["scaleIn"]["count"]){
            // Check count minimum size
            int scaleCount = configData["scaleIn"]["count"].as<int>() >= 2 ? configData["scaleIn"]["count"].as<int>() : 2;
            optDimension += scaleCount - 1;
        }else{
            optDimension += 1;

        }
    } 


    Eval::set_dim_in(optDimension);
    std::cout << "Number of optimized parameters: " << optDimension << std::endl;

    return optDimension;
}

// Main function
int main(int argc, char **argv)
{
    // Local variables
    std::string trainDataFile, evalDataFile, testDataFile;
    MatrixBO trainData, trainLabel, evalData, evalLabel, testData;
    std::string trainLabelFile, evalLabelFile, testLabelFile;
    std::string samplingTrainFile, samplingValFile, samplingTestFile;
    Eigen::VectorXd bestSample;
    // Read configuration file
    std::string yamlFile;
    std::string inputFolder;
    std::string computationType;
    if (argc > 2)
    {
        computationType = std::string(argv[1]);

        inputFolder = std::string(argv[2]);
        
        if (computationType == "compute")
        {
            yamlFile = inputFolder + std::string("/optimal.yaml");
        }else{
            yamlFile = inputFolder;
        }

        std::cout << "Configuration file: " << yamlFile << std::endl;
        configData = YAML::LoadFile(yamlFile);
    }
    else
    {
        std::cout << "Usage: test computationType yamlFile/inputFolderName [outputFolderName]" << std::endl;
        return -1;
    }

    // Read folder file
    std::string NameFolder;
    if (argc > 3)
    {
        NameFolder = std::string(argv[3]);
        std::cout << "new experiment folder: " << NameFolder << std::endl;
    }
    else
    {
        NameFolder = std::string("");
    }

    // Load python interpreter if necessary
#ifdef USE_PYBIND
    if (configData["Problem_Definition"]["type"].as<std::string>() == "External")
    {
        pybind11::initialize_interpreter();
    }    
#endif

    // Set openmp threads
#ifdef _OPENMP
    int ompThreads = 1;
    if (configData["eigen_threads"])
    {
        ompThreads = configData["eigen_threads"].as<int>(); 
    }
    Eigen::setNbThreads(ompThreads);
    std::cout << "Eigen using " << Eigen::nbThreads() << " threads" << std::endl;
#endif

    // Set optimization dimension
    int optDimension = set_optDimension();
    bestSample.resize(optDimension);
    
    // Set random sampling against optDimension
    if (optDimension > 3)
    {
        Params::init_randomsampling::set_samples(10 + 3 * (optDimension - 3));
        std::cout << "Number of random samples:" << 10 + 3 * (optDimension - 3) << std::endl;
    }
    else
    {
        Params::init_randomsampling::set_samples(10);
        std::cout << "Number of random samples:" << 10 << "\n";
    }

    // Set ealy stop threshold
    if (configData["early_stop"])
    {
        Params::stop_maxfitness::set_threshold(configData["early_stop"].as<float>());
    }
    else
    {
        Params::stop_maxfitness::set_threshold(100);
    }

    // Check numbers of datasets
    int nDatasets = 1;
    if (configData["num_datasets"])
    {
        nDatasets = configData["num_datasets"].as<int>();
    }

    // If multiple Dataset, create a "Main" Folder for clarity
    std::string MainFolder = "";
    if (nDatasets > 1)
    {
        boost::filesystem::create_directory(NameFolder);
        MainFolder = NameFolder + "/";
    }

    // Set problem type
    std::string problemType = configData["Problem_Definition"]["type"].as<std::string>();

    // Control execution
    for (int dataIter = 1; dataIter <= nDatasets; dataIter++)
    {
        // Set dataset iteration
        std::cout << "Executing dataset " << dataIter << std::endl;
        Eval::set_data_it(dataIter);
        std::string SpecificFolder;
        SpecificFolder = MainFolder + NameFolder + std::to_string(dataIter);


        // Load training data from files
        trainDataFile = replace_tag(configData["train_data"].as<std::string>(), dataIter);
        trainLabelFile = replace_tag(configData["train_label"].as<std::string>(), dataIter);
        series.load_data(trainDataFile, trainLabelFile, "train");
        trainData = series.get_data("train", "data");
        trainLabel = series.get_data("train", "label");

        // Load validation data from files
        evalDataFile = replace_tag(configData["eval_data"].as<std::string>(), dataIter);
        evalLabelFile = replace_tag(configData["eval_label"].as<std::string>(), dataIter);
        series.load_data(evalDataFile, evalLabelFile, "valid");
        evalData = series.get_data("valid", "data");
        evalLabel = series.get_data("valid", "label");

        // Check classes in validation for classification tasks
        if((evalLabel.maxCoeff() > trainLabel.maxCoeff()) & problemType == "Classification")
        {
            std::cout << "Error! Validation labels contain more classes than Training labels" << std::endl;
            return -1;
        }

        // Load Sampling vectors
        if (configData["train_sampling"])
        {
            samplingTrainFile = replace_tag(configData["train_sampling"].as<std::string>(), dataIter);
            samplingValFile = replace_tag(configData["eval_sampling"].as<std::string>(), dataIter);
            load_csv(samplingTrainFile, samplingTrain);
            load_csv(samplingValFile, samplingVal);
        }
        else
        {
            samplingTrain = MatrixBO::Constant(trainData.rows(), 2, 1);
            samplingVal = MatrixBO::Constant(evalData.rows(), 2, 1);
        }

        // Control compute vs training
        using Kernel_t = kernel::MaternFiveHalves<Params>;
        using Mean_t = mean::Data<Params>;
        using GP_t = model::GP<Params, Kernel_t, Mean_t>;
        using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>, MaxFitness<Params>>;
        using Acqui_t = acqui::UCB<Params, GP_t>;
        using Init_t = ParSampler<Params>;
        bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>, stopcrit<Stop_t>, initfun<Init_t>> boptimizer;
        
        // Reload config data TODO fix it
        configData = YAML::LoadFile(yamlFile);
        if (computationType == "train")
        {
            //esn_compute(trainData, "optimal");
            //ArrayBO cose = esn_caller("Laserss1", series, nLayers);
            std::cout << "Optimization Start" << std::endl;
            // Set Limbo parameters
            //using Kernel_t = kernel::MaternFiveHalves<Params>;
            //using Mean_t = mean::Data<Params>;
            
            // Sample time start
            time_stats.tStart = high_resolution_clock::now();

            // Check for memory optimization
            double penal = 0;
            if (configData["Memory_Optimization_Coeff"])
            {
                penal = configData["Memory_Optimization_Coeff"].as<double>();
            }
            // Run the evaluation
            if (penal > 0)
            {
                std::cout << "Memory Optimizer START" << std::endl;
                boptimizer.optimize(Eval(), AggregatorMemoryOpt());
            }
            else
            {
                boptimizer.optimize(Eval(), AggregatorNaive());
            }

            save_time_samples(boptimizer.res_dir(), time_stats);

            // Select best sample
            std::cout << "Best sample: " << boptimizer.best_sample().transpose() << std::endl;
            std::string opt_config_file = boptimizer.res_dir() + "/optimal.yaml";
            save_config(opt_config_file, configData, boptimizer.best_sample());
            bestSample = boptimizer.best_sample();

            // Save optimization Result

            std::ofstream final_out;
            final_out.open(boptimizer.res_dir() + "/aggregated_observations.dat", std::ofstream::app);
            final_out << boptimizer.current_iteration() << " ";
            final_out << std::fixed;
            final_out.close();
            final_out.open(boptimizer.res_dir() + "/samples.dat", std::ofstream::app);
            final_out << boptimizer.current_iteration() << " ";
            final_out << std::fixed << std::setprecision(12) << boptimizer.best_sample().transpose() << std::endl;
            final_out << std::fixed;
            final_out.close();
        }
        else if(computationType == "compute") 
        {
            for (int i = 0; i < optDimension; ++i)
            {
                bestSample(i) = configData["x"][i].as<double>();
            }         
        }
        // Create validation data
        MatrixBO fullTrainData = cat_matrix(trainData, evalData);
        MatrixBO fullTrainLabel = cat_matrix(trainLabel, evalLabel);
        MatrixBO fullTrainSampling = cat_matrix(samplingTrain, samplingVal);

        // Copy it in data storage
        series.copy_data(fullTrainData, fullTrainLabel, "train");

        // Load test data from files
        testDataFile = replace_tag(configData["test_data"].as<std::string>(), dataIter);
        testLabelFile = replace_tag(configData["test_label"].as<std::string>(), dataIter);
        series.load_data(testDataFile, testLabelFile, "test");
        testData = series.get_data("test", "data");

        // Load test sampling
        if (configData["test_sampling"])
        {
            samplingTestFile = replace_tag(configData["test_sampling"].as<std::string>(), dataIter);
            load_csv(samplingTestFile, samplingTest);
        }
        else
        {
            samplingTest = MatrixBO::Constant(testData.rows(), 2, 1);
        }
        
        // Set test sampling
        int nWashout = parse_config<int>("washout_sample", -1, 0, bestSample, configData, 10);
        series.set_sampleArray(fullTrainSampling, nWashout, true, problemType, "train");
        bool init_flag = fullTrainSampling(fullTrainSampling.rows() - 1, 1) == 0;
        series.set_sampleArray(samplingTest, nWashout, init_flag, problemType, "valid");

        // Obtain fitness on test data
        ArrayBO fitness;
 
         // Save final fitness
        std::ofstream final_out;
        final_out.open(boptimizer.res_dir() + "/aggregated_observations.dat", std::ofstream::app);
        
        fitness = esn_caller(bestSample, configData, boptimizer.res_dir().c_str(), series, true, computationType, inputFolder);   
        
        final_out << std::fixed << std::setprecision(12) << fitness.transpose() << std::endl;
        final_out.close();

        // Set Folder Name
        if (!NameFolder.empty())
        {
            int success = rename(boptimizer.res_dir().c_str(), SpecificFolder.c_str());
            if (success != 0)
            {
                int k = 1;
                while (success != 0)
                {
                    success = rename(boptimizer.res_dir().c_str(), (SpecificFolder + "-" + std::to_string(k)).c_str());
                    k += 1;
                    if(k > 5){
                        std::cout << "Failed to rename, quitting." << std::endl;
                        break;
                    }
                }
            }
        }
    }

    // Unload python interpreter if necessary
#ifdef USE_PYBIND
    if (configData["Problem_Definition"]["type"].as<std::string>() == "External")
    {
        pybind11::finalize_interpreter();
    }    
#endif

    return 0;
}
