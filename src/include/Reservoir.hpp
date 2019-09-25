#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include <random>
#include <chrono>
#include <utility>
#include <unsupported/Eigen/SparseExtra>
#include <spectra/GenEigsSolver.h>
#include <spectra/MatOp/SparseGenMatProd.h>
#include <spectra/SymEigsSolver.h>
#include <exception>
#include "EchoBay.hpp"
#include "yaml-cpp/yaml.h"
#include "Eigen/StdVector"

#ifdef USE_TBB
#include <tbb/mutex.h>
#endif

#include "ConfigUtils.hpp"
#if !defined(ESP_PLATFORM)
#include "IOUtils.hpp"
#endif
#include <iomanip> // Due to print Params


/**
 * Struct defining the parametrization of a ESN layer
 * 
 */
typedef struct {
    /** Reservoir size */
    int Nr;
    /** Reservoir density */
    double density;
    /** Desired spectral radius */
    double desiredRho;
    /** Leaky integration factor */
    double leaky;
    /** Input scaling parameter */
    std::vector<double> scaleInput; // Not best practice
    /** Extra Parameter for Edges (Small World topology) and Jumps (Cyclic Reservoir topology) */
    int edgesJumps;
} layerParameter;

/** Map different topologies of Reservoir */
static std::map<int, std::string> ESNTypes = {{0, "Random Reservoir"}, {1, "Small World"}, {2, "Cyclic Reservoir Jump"}};

/* Useful lambdas TODO find a better place*/
auto sumNr = [](int sum, const layerParameter& curr){return sum + curr.Nr;};
auto sumNrSWT = [](int sum, const ArrayI curr){return sum + curr.size();};
namespace EchoBay
{
    /**
     * Implementation of Echo State Network Reservoir unit
     * 
     * Reservoir is defined by the actual reservoir Wr, a input matrix Win and a non-linear state.
     * Initialization of the Reservoir is defined by a high-level configuration structure which supports also
     * differentiated input scaling, particular topologies, and deep networks.
     * 
     */
    class Reservoir
    {
        public:
        Reservoir(const int nLayers = 1, const int Nu = 2, const int type = 0);
        ~Reservoir();
        SparseBO init_Wr(const int Nr, const double density, const double scalingFactor, const double leaky, const int extraParams);

        // Initialization of Network if type is not available
        void init_network();
        void init_WinLayers();
        void init_WrLayers();
        void init_stateMat();

        // SMALL WORLD TOPOLOGY
        void init_WinLayers_swt();

        // CRJ TOPOLOGY
        void init_WinLayers_crj();

        // IO network
#if !defined(ESP_PLATFORM)
        // Load network files if available
        void load_network(  const std::string &folder);
        void load_WinLayers(const std::string &folder);
        void load_WrLayers( const std::string &folder);
        void load_stateMat( const std::string &folder);
#endif

        // LayerConfig functions
        void init_LayerConfig(const Eigen::VectorXd &optParams, const YAML::Node confParams);
        void init_LayerConfig(std::vector<stringdouble_t> paramValue);
        std::vector<layerParameter> get_LayerConfig() const;
        int get_nLayers() const;
        int get_fullNr(const int layer = -1) const;
        // SWT structures
        std::vector<ArrayI> get_WinIndex() const;
        std::vector<ArrayI> get_WoutIndex() const;
        int get_NrSWT(const int layer = -1) const;

        // Utils for Reservoir
        int get_ReservoirType() const;
        //std::vector<int> pick(int Nr, int k); 
        ArrayI pick(int Nr, int k); 
        std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen);

        // Print parameters
        void print_params(const int nLayers, const int nWashout, const double lambda);
        floatBO return_net_dimension(const YAML::Node confParams) const;

        // Reservoir elements
        std::vector<MatrixBO> WinLayers;
        std::vector<SparseBO> WrLayers;
        std::vector<ArrayBO> stateMat;

        private:
        void wr_random_fill(SparseBO &Wr, int Nr,  int active_units, std::unordered_set< std::pair<int,int>, pair_hash> &valid_idx_hash);
        void wr_swt_fill(   SparseBO &Wr, int Nr,  int edges,        std::unordered_set<std::pair<int, int>, pair_hash> &valid_idx_hash);
        void wr_crj_fill(   SparseBO &Wr, int Nr,  int jumps,        std::unordered_set<std::pair<int, int>, pair_hash> &valid_idx_hash);
        floatBO wr_scale_radius(SparseBO &Wr, const double scalingFactor);

        // Internal variables
        int _nLayers;
        int _Nu;
        std::vector<layerParameter> _layerConfig;
        int _type; // 0 for Random, 1 for SWT, 2 for CRJ
        // SWT Structures
        std::vector<ArrayI> _WinIndex;
        std::vector<ArrayI> _WoutIndex;
        std::string _reservoirInizialization = "radius";        
    };

} // namespace EchoBay

#endif