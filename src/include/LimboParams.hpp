#include <iostream>
#include <fstream>
#include <chrono>
#include <limbo/limbo.hpp>

#include <limbo/serialize/text_archive.hpp>
#include <limbo/init/random_sampling.hpp>

#include "EigenConfig.hpp"
#include "DataStorage.hpp"

// bad global variables
MatrixBO samplingTrain, samplingVal, samplingTest;
EchoBay::DataStorage series;

typedef struct
{
    std::chrono::high_resolution_clock::time_point tStart;
    std::vector<std::chrono::high_resolution_clock::time_point> tSamples;
} time_samples;

YAML::Node configData;
time_samples time_stats;

void save_time_samples(std::string folder, time_samples input)
{
    std::ofstream outputFile(folder + "/tSamples.dat", std::ios::out);
    double sample;
    for (size_t i = 0; i < input.tSamples.size(); i++)
    {
        sample = std::chrono::duration_cast<std::chrono::nanoseconds>(input.tSamples[i] - input.tStart).count() / 1e+09;
        outputFile << i + 1 << "\t" << sample << std::endl;
    }
    outputFile.close();
}

struct Params
{
    struct bayes_opt_boptimizer : public limbo::defaults::bayes_opt_boptimizer
    {
    };

// depending on which internal optimizer we use, we need to import different parameters
#ifdef USE_NLOPT
    struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd
    {
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public limbo::defaults::opt_cmaes
    {
    };
#else
    struct opt_gridsearch : public limbo::defaults::opt_gridsearch
    {
        BO_PARAM(int, bins, 20);
    };
#endif

    struct kernel : public limbo::defaults::kernel
    {
        BO_PARAM(double, noise, 0.001);
    };

    struct bayes_opt_bobase : public limbo::defaults::bayes_opt_bobase
    {
    };

    struct kernel_maternfivehalves : public limbo::defaults::kernel_maternfivehalves
    {
    };

    struct init_randomsampling : public limbo::defaults::init_randomsampling
    {
        BO_DYN_PARAM(int, samples);
    };

    struct init_gridsampling : public limbo::defaults::init_gridsampling
    {
        BO_PARAM(int, bins, 10);
    };

    struct stop_maxiterations : public limbo::defaults::stop_maxiterations
    {
        BO_PARAM(int, iterations, 40);
    };
    struct stop_maxfitness
    {
        BO_DYN_PARAM(float, threshold);
    };

    // we use the default parameters for acqui_ucb
    struct acqui_ucb : public limbo::defaults::acqui_ucb
    {
    };
    // Added Expected Improvement because UCB seems to jump too much between samples
    struct acqui_ei : public limbo::defaults::acqui_ei
    {
    };
};

template <typename Params>
struct MaxFitness
{
    MaxFitness() {}

    template <typename BO, typename AggregatorFunction>
    bool operator()(const BO &bo, const AggregatorFunction &afun)
    {
        bool stop = afun(bo.best_observation(afun)) > Params::stop_maxfitness::threshold();
        if (stop && bo.current_iteration() >= 1)
        {
            std::cout << "stop caused by Max predicted value reached. Threshold: "
                      << Params::stop_maxfitness::threshold()
                      << " max observations: " << afun(bo.best_observation(afun)) << std::endl;
            return true;
        }
        return false;
    }
};
template <typename Params>
struct NoVariation
{
    NoVariation() {}

    template <typename BO, typename AggregatorFunction>
    bool operator()(const BO &bo, const AggregatorFunction &afun)
    {
        double value = 100;
    if (bo.current_iteration() > 3)
    {
        value = 0;

        auto vec = bo.samples();
        std::cout << vec.size() << std::endl;

        //int posix = 0;
        for (size_t i = vec.size()-3; i < vec.size()-1; ++i)
        {
            //posix = i%3;
            //Points.row(posix) = vec[i](0);
            //std::cout << vec[i] << std::endl;
            value += (vec[i]-vec[i+1]).norm();
        }
        value /= 3;
        std::cout << "norm is " << value << std::endl;

    }
    if (value < 1E-4)
    {
        return true;
    }else
    {
        return false;
    }


    }
};

struct AggregatorMemoryOpt {
    using result_type = double;
    double operator()(const Eigen::VectorXd& x) const
    {
        double result = x(0) - x(1);
        return result;
    }
};

struct AggregatorNaive {
    using result_type = double;
    double operator()(const Eigen::VectorXd& x) const
    {
        return x(0);
    }
};

template <typename Params>
struct ParSampler
{
    template <typename StateFunction, typename AggregatorFunction, typename Opt>
    void operator()(const StateFunction &seval, const AggregatorFunction &, Opt &opt) const
    {
        // Set number of threads
        int nbThreads = tbb::task_scheduler_init::default_num_threads();
        if (configData["sample_threads"])
        {
            if (configData["sample_threads"].as<int>() <= tbb::task_scheduler_init::default_num_threads())
            {
                nbThreads = configData["sample_threads"].as<int>();
            }
        }

        // Initialize TBB
        tbb::task_scheduler_init init(nbThreads);
        if (init.is_active())
        {
            std::cout << "Using " << nbThreads << " threads" << std::endl;
        }
        else
        {
            std::cout << "TBB inactive. Using 1 thread" << std::endl;
        }

        // Grid Sampling
        //_explore(0, seval, Eigen::VectorXd::Constant(StateFunction::dim_in(), 0), opt);
        // To use Grid Sampling, Uncomment this line and comment the next paragraph.
        // Put to 1 the iterations in stop_max_iterations

        //Random Sampling
        limbo::tools::par::loop(0, Params::init_gridsampling::bins(), [&](size_t i)
        {
            auto newSample = limbo::tools::random_vector(StateFunction::dim_in(), Params::bayes_opt_bobase::bounded());
            opt.eval_and_add(seval, newSample);
        });
    }
    // Grid Sampling Function
    template <typename StateFunction, typename Opt>
    void _explore(int dim_in, const StateFunction& seval, const Eigen::VectorXd& current,
        Opt& opt) const
    {
        limbo::tools::par::loop(0, (double)Params::init_gridsampling::bins() , [&](size_t x)
        {
            auto z =  (double)x / (double)Params::init_gridsampling::bins();
            Eigen::VectorXd point = current;
            point[dim_in] = z;
            std::cout << z << std::endl;
            if (dim_in == current.size() - 1) {
                opt.eval_and_add(seval, point);
            }
            else {
                _explore(dim_in + 1, seval, point, opt);
            }

        });
    }
};