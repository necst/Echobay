#include "Reservoir.hpp"
#include "Eigen/StdVector"
#include "Quantizer.hpp"

#ifdef USE_TBB
#ifndef COUT_MU
tbb::mutex cout_mutex;
#endif
#endif

using namespace std::chrono;
using namespace Eigen;
using namespace Spectra;


class WrException: public std::exception
{
  virtual const char* what() const throw()
  {
    return "Wr: bad_alloc";
  }
} WrEx;

/**
 * @brief Construct a Reservoir class with a given topology, number of layers and inputs
 * 
 * @param nLayers Number of layers in deep reservoirs (= 1 in shallow networks)
 * @param Nu Number of input channels
 * @param type Topology of the network see also EchoBay::esn_config(const Eigen::VectorXd &optParams, YAML::Node confParams, int Nu, const std::string folder)
 * for details
 */
EchoBay::Reservoir::Reservoir(const int nLayers, const int Nu, const int type)
{
    // Save layers
    _nLayers = nLayers;
    // Save input
    _Nu = Nu;
    // Save type
    _type = type;
    // Reserve vectors
    WinLayers.reserve(_nLayers);
    WrLayers.reserve(_nLayers);
    stateMat.reserve(_nLayers);
    _layerConfig.reserve(_nLayers);
}

/**
 * @brief Destroy the Reservoir object cleaning internal layers
 * 
 */
EchoBay::Reservoir::~Reservoir()
{
    // Free vectors
    _layerConfig.clear();
    _layerConfig.shrink_to_fit();
    WinLayers.clear();
    WinLayers.shrink_to_fit();
    WrLayers.clear();
    WrLayers.shrink_to_fit();
    stateMat.clear();
    stateMat.shrink_to_fit();
}

/**
 * @brief Initialize the reservoir according to internal topology
 * 
 */
void EchoBay::Reservoir::init_network()
{
    // Initialize Win
    switch(_type)
    {
        case 0: this->init_WinLayers();
                break;
        case 1: this->init_WinLayers_swt();
                break;
        case 2: this->init_WinLayers_crj();
                break;
    }

    // Initialize Wr
    this->init_WrLayers();

    // Initialize stateMat
    this->init_stateMat();
}

/**
 * @brief Initialize random input layers
 * 
 */
void EchoBay::Reservoir::init_WinLayers()
{
    // Reset Win
    WinLayers.clear();
    // Determine the input scaling rule
    int countScale = _layerConfig[0].scaleInput.size(); // TO DO: Not correct way for determining countscale

    // Homogeneous scaling
    if (countScale == 1)
    {        
        WinLayers.push_back(_layerConfig[0].scaleInput[0] * MatrixBO::Random(_layerConfig[0].Nr, _Nu));
    }
    else
    {
        // Differentiated input scaling
        MatrixBO Win(_layerConfig[0].Nr, _Nu);
        for (int cols = _Nu - 1; cols >= 0; cols--)
        {
            Win.col(cols) = _layerConfig[0].scaleInput[cols] * MatrixBO::Random(_layerConfig[0].Nr, 1);
        }
        WinLayers.push_back(Win);
    }
    // Fill subsequent layers
    for (int i = 1; i < _nLayers; ++i)
    {
        WinLayers.push_back(_layerConfig[i].scaleInput[0] * MatrixBO::Random(_layerConfig[i].Nr, _layerConfig[i - 1].Nr + 1));   
    }
}

/**
 * @brief Initialize input layers according to Small World Topology selection
 * 
 */
void EchoBay::Reservoir::init_WinLayers_swt()
{
    // Reset Win
    WinLayers.clear();

    // Determine the input scaling rule
    int countScale = _layerConfig[0].scaleInput.size();//-1;
    std::vector<ArrayI> NIindex = _WinIndex;
    std::vector<ArrayI> NOIndex = _WoutIndex;
    int NiCount = NIindex[0].size();
    int NoCount;

    // Homogeneous scaling
    if (countScale == 1)
    {
        MatrixBO temp = MatrixBO::Zero(_layerConfig[0].Nr, _Nu);
        temp.block(0,0,NiCount, _Nu) = _layerConfig[0].scaleInput[0] * MatrixBO::Random(NiCount, _Nu);

        WinLayers.push_back(temp);
    }
    else
    {
        // Differentiated input scaling
        MatrixBO Win(_layerConfig[0].Nr, _Nu);
        for (int cols = _Nu - 1; cols >= 0; cols--)
        {
            MatrixBO temp = MatrixBO::Zero(_layerConfig[0].Nr, 1);
            // Select a block of inputs
            temp.block(0,0,NiCount, 1) =  _layerConfig[0].scaleInput[cols] * MatrixBO::Random(NiCount, 1);
            Win.col(cols) = temp;
        }
        
        WinLayers.push_back(Win);
    }
    // Fill subsequent layers
    for (int i = 1; i < _nLayers; ++i)
    {
        MatrixBO temp = MatrixBO::Zero(_layerConfig[i].Nr, _layerConfig[i-1].Nr + 1);
        NiCount = NIindex[i].size();
        NoCount = NOIndex[i-1].size();
        temp.block(0,floor(_layerConfig[i-1].Nr/2),NiCount, NoCount) = _layerConfig[i].scaleInput[0] * MatrixBO::Random(NiCount, NoCount);
        temp.block(0,_layerConfig[i-1].Nr,NiCount, 1) = _layerConfig[i].scaleInput[0] * MatrixBO::Random(NiCount, 1);
        WinLayers.push_back(temp);
        // MatrixBO temp(_layerConfig[i].Nr, _layerConfig[i-1].Nr + 1);
        // WinLayers.push_back(_layerConfig[i].scaleInput[0] * MatrixBO::Random(_layerConfig[i].Nr, _layerConfig[i - 1].Nr + 1));       
    }
}

/**
 * @brief Initialize input layers according to Cyclic Reservoir Jumps selection
 * 
 */
void EchoBay::Reservoir::init_WinLayers_crj()
{
    // Reset Win
    WinLayers.clear();

    // Determine the input scaling rule
    int countScale = _layerConfig[0].scaleInput.size();//-1;

    // Initialize random engine
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::bernoulli_distribution probability(0.5);

    // Determine Win sign
    auto opSign = [&](void){return probability(gen) ? 1 : -1;};
    MatrixBO temp = MatrixBO::NullaryExpr(_layerConfig[0].Nr, _Nu, opSign); // Apply value using lambda

    // Homogeneous scaling
    if (countScale == 1)
    {
        WinLayers.push_back(_layerConfig[0].scaleInput[0] * temp);
    }
    else
    {
        // Differentiated input scaling
        MatrixBO Win(_layerConfig[0].Nr, _Nu);
        for (int cols = _Nu - 1; cols >= 0; cols--)
        {
            Win.col(cols) = _layerConfig[0].scaleInput[cols] * temp.col(cols);
        }
        WinLayers.push_back(Win);
    }
    // Fill subsequent layers
    for (int i = 1; i < _nLayers; ++i)
    {
        temp = MatrixBO::NullaryExpr(_layerConfig[i].Nr, _layerConfig[i - 1].Nr + 1, opSign);
        WinLayers.push_back(_layerConfig[i].scaleInput[0] * temp);
    }
}

/**
 * @brief Reset state matrix
 * 
 */
void EchoBay::Reservoir::init_stateMat()
{
    // Clear state vector
    stateMat.clear();
    for (const auto& layer: _layerConfig)
    {
        stateMat.push_back(ArrayBO::Zero((layer.Nr)));
    }
}

/**
 * @brief Initialize the weights of the Reservoir
 * 
 */
void EchoBay::Reservoir::init_WrLayers()
{
    // Clear Wr vector
    WrLayers.clear();
    // Fill layers
    for(const auto& layer: _layerConfig)
    {
        WrLayers.push_back(init_Wr(layer.Nr, layer.density, layer.desiredRho, layer.leaky, layer.edgesJumps));
    }
}

#if !defined(ESP_PLATFORM)
/**
 * @brief Load the whole Reservoir from files
 * 
 * @param folder Path to the folder containing the files
 */
void EchoBay::Reservoir::load_network(const std::string &folder)
{
    // Initialize Win
    this->load_WinLayers(folder);

    // Initialize Wr
    this->load_WrLayers(folder);

    // Initialize stateMat
    this->load_stateMat(folder);
}

/**
 * @brief Load input layers from files
 * 
 * @param folder Path to the folder containing the files
 */
void EchoBay::Reservoir::load_WinLayers(const std::string &folder)
{

    MatrixBO Win;
    std::string tempName;
    for (int i = 0; i < _nLayers; ++i)
    {
        tempName = folder + "/Win_eigen" + std::to_string(i) + ".mtx";
        read_matrix(tempName, Win);
        WinLayers.push_back(Win);
    }
}

/**
 * @brief Load Reservoir's weight layers from files
 * 
 * @param folder Path to the folder containing the files
 */
void EchoBay::Reservoir::load_WrLayers(const std::string &folder)
{
    SparseMatrix<floatBO, 0x1> Wr;
    std::string tempName;
    for (int i = 0; i < _nLayers; ++i)
    {
        tempName = folder + "/Wr_eigen" + std::to_string(i) + ".mtx";
        loadMarket(Wr, tempName);
        WrLayers.push_back(Wr);
    }
}

/**
 * @brief Load Reservoir's states from files
 * 
 * @param folder Path to the folder containing the files
 */
void EchoBay::Reservoir::load_stateMat(const std::string &folder)
{
    // First layer
    ArrayBO state;
    std::string tempName;
    for (int i = 0; i < _nLayers; ++i)
    {
        tempName = folder + "/State_eigen" + std::to_string(i) + ".mtx";
        read_matrix(tempName, state);
        stateMat.push_back(state);
    }
}
#endif

/**
 * @brief Fill the reservoir with random initialization
 * 
 * @param Wr Eigen SparseMatrix of the weights
 * @param Nr Layer dimension
 * @param active_units Number of active units to be allocated randomly
 * @param valid_idx_hash [Not used] keep track of the active units indexes
 */
void EchoBay::Reservoir::wr_random_fill(SparseBO &Wr, int Nr, int active_units, std::unordered_set<std::pair<int, int>, pair_hash> &valid_idx_hash)
{
    // Initialize random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> values(-1.0, 1.0);
    std::uniform_real_distribution<double> indexes(0.0, 1.0);

    std::vector<TripletBO> tripletList;
    tripletList.reserve(active_units);

    std::pair<UOMIterator, bool> result;

    // Fill Wr matrix
    int i, j, k;
    floatBO value;
    for (k = 0; k < active_units;)
    {
        i = floor(indexes(gen) * Nr);
        j = floor(indexes(gen) * Nr);

        value = values(gen);

        // Inserting an element through value_type
        result = valid_idx_hash.insert({i, j});
        if (result.second == true)
        {
            tripletList.push_back(TripletBO(i, j, value));
            k++;
        }
    }
    try
    {
        Wr.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    catch (const std::exception &e)
    {
        throw WrEx;
    }
}

/**
 * @brief Fill the reservoir with a Small World Topology
 * 
 * @param Wr Eigen SparseMatrix of the weights
 * @param Nr Layer dimension
 * @param edges Number of edges connected to each unit TODO add reference
 * @param valid_idx_hash [Not used] keep track of the active units indexes
 */
void EchoBay::Reservoir::wr_swt_fill(SparseBO &Wr, int Nr,  int edges, std::unordered_set<std::pair<int, int>, pair_hash> &valid_idx_hash)
{
    // Initialize random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> values(-1.0, 1.0);
    std::bernoulli_distribution probability(0.1);
    std::uniform_real_distribution<double> indexToReassign(0, Nr-1);

    // Allocate triplet list
    std::vector<TripletBO> tripletList;
    std::pair<UOMIterator, bool> result;

    // Fill Wr matrix
    int i, j;
    floatBO value;

    // Check on maximum number of edges
    if(edges >= ceil(Nr/4))
    {
        std::cout << "WARNING! Limiting SWT Edges to Nr/4 to avoid bad rewiring." << std::endl;
        edges = ceil(Nr/4);
    }

    // Allocate fixed connections
    for ( i = 0; i < Nr; ++i)
    {
        for (j = 1  ; j <= edges; ++j)
        {
            // Pick value
            value = values(gen);

            int tempIndex = i + j;
            tempIndex = tempIndex > Nr-1 ? tempIndex - Nr : tempIndex;
            tripletList.push_back(TripletBO(i, tempIndex, value));
            tripletList.push_back(TripletBO( tempIndex, i, value));
        }
    }

    // Set initial Wr
    try
    {
        Wr.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    catch (const std::exception &e)
    {
        throw WrEx;
    }

    // Move edges according to rewiring probability
    for ( j = 1; j <= edges; ++j)
    {
        // int tempIndexLeft = i - j;
        // tempIndexLeft = tempIndexLeft < 0 ? tempIndexLeft + Nr : tempIndexLeft;
        for ( i = 0; i < Nr; ++i )
        {
            int tempIndexRight = i + j;
            tempIndexRight = tempIndexRight > Nr-1 ? tempIndexRight - Nr : tempIndexRight;
            if(probability(gen)){
                Wr.coeffRef(i,tempIndexRight) = 0;
                Wr.coeffRef(tempIndexRight,i) = 0;
                int newConnection = ceil(indexToReassign(gen));

                while(Wr.coeffRef(i, newConnection))
                {
                    while (Wr.coeffRef(i,newConnection) != 0)
                    {
                        newConnection = ceil(indexToReassign(gen));
                    }
                }
                // Move connection
                value = values(gen);
                if (!Wr.coeffRef(i, newConnection))
                {
                    Wr.coeffRef(i, newConnection) = value;
                    Wr.coeffRef(newConnection,i) = value;
                }else{
                    Wr.insert(i, newConnection) = value;
                    Wr.insert(newConnection, i) = value;
                }
            }
        }
    }
}

/**
 * @brief Fill the reservoir with a Cyclic Reservoir Jump topology
 * 
 * @param Wr Eigen SparseMatrix of the weights
 * @param Nr Layer dimension
 * @param jumps Number of jumps in the connections
 * @param valid_idx_hash [Not used] keep track of the active units indexes
 */
void EchoBay::Reservoir::wr_crj_fill(SparseBO &Wr, int Nr, int jumps, std::unordered_set<std::pair<int, int>, pair_hash> &valid_idx_hash)
{
    // Initialize random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> values(-1.0, 1.0);
    std::bernoulli_distribution probability(0);
    std::uniform_real_distribution<double> indexToReassign(0, Nr-1);

    // Allocate triplet list
    std::vector<TripletBO> tripletList;
    std::pair<UOMIterator, bool> result;

    // Fill Wr matrix
    int i;
    floatBO value;
    // Create Circle
    value = values(gen);
    tripletList.push_back(TripletBO( 0, Nr-1, value));
    //tripletList.push_back(TripletBO( Nr-1, 0, value));

    // Initial fill
    for(i = 0; i < Nr-1; i++){
        tripletList.push_back(TripletBO( i+1, i, value));
        //tripletList.push_back(TripletBO( i, i+1, value));
    }

    // Create Jumps
    value = values(gen);
    if (jumps > 0)
    {
        for (i = 0; i < Nr; i = i + jumps){
            int tempindex = i + jumps;
            tempindex = tempindex > Nr-1 ? tempindex - Nr : tempindex;
            tripletList.push_back(TripletBO( i, tempindex, value));
            tripletList.push_back(TripletBO( tempindex, i, value));
        }
    }

    // Allocate Wr
    try
    {
        Wr.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    catch (const std::exception &e)
    {
        throw WrEx;
    }
}


/**
 * @brief Calculate Spectral Radius of weight matrix
 * 
 * @param Wr Eigen SparseMatrix of the weights
 * @param scalingFactor Additional custom scaling factor
 * @return floatBO Scale radius
 */
floatBO EchoBay::Reservoir::wr_scale_radius(SparseBO &Wr, const double scalingFactor)
{
    // Spectra solver for eigenvalues
    SparseGenMatProd<floatBO, 0x1> op(Wr);

    int param = 6 > Wr.rows() ? Wr.rows() : 6;
    VComplexBO evalues;
    floatBO scale_radius;
    int nconv;

    // General case with Nr > 2
    if(Wr.rows() > 2)
    {
        do
        {
            GenEigsSolver<floatBO, LARGEST_MAGN, SparseGenMatProd<floatBO, 0x1>> eigs(&op, 1, param);
            eigs.init();
            nconv = eigs.compute();
            param = (param + 10 > Wr.rows()) ? Wr.rows() : param + 10;
            if (nconv > 0){ //ok == SUCCESSFUL)
                evalues = eigs.eigenvalues();
            }
        } while (nconv < 1); //ok == NOT_CONVERGING);

        // Rescale Wr matrix with desired rho value
        floatBO spectral_radius = sqrt(pow(std::real(evalues[0]), 2) + pow(std::imag(evalues[0]), 2));
        scale_radius = scalingFactor / spectral_radius;
    }
    else // 2-by-2 matrix
    {
        // Find trace and determinant
        floatBO trace = Wr.coeff(0,0) + Wr.coeff(1,1);
        floatBO determinant = Wr.coeff(0,0)*Wr.coeff(1,1) - Wr.coeff(1,0)*Wr.coeff(0,1);

        // Calculate the two eigenvalues
        VComplexBO eigens = VComplexBO::Zero(2,1);
        eigens(0) = (trace + sqrt(trace*trace - 4.0*determinant))/2.0;
        eigens(1) = (trace - sqrt(trace*trace - 4.0*determinant))/2.0;
        floatBO magEigen[2] = {0};
        magEigen[0] = sqrt(eigens(0).real()*eigens(0).real() + eigens(0).imag()*eigens(0).imag());
        magEigen[1] = sqrt(eigens(1).real()*eigens(1).real() + eigens(1).imag()*eigens(1).imag());
        scale_radius = magEigen[0] > magEigen[1] ? scalingFactor/magEigen[0] : scalingFactor/magEigen[1];
    }
    return scale_radius;
}

/**
 * @brief Initialize the Reservoir's weights according to user-defined parameters
 * 
 * @param Nr Reservoir size
 * @param density Reservoir density
 * @param scalingFactor Custom scaling factor TODO add reference
 * @param leaky Leaky factor
 * @param extraParam Extra parameter defining edges (SW topology) or jumps (CRJ topology) 
 * @return SparseBO Scaled Wr
 */
SparseBO EchoBay::Reservoir::init_Wr(const int Nr, const double density,
                                     const double scalingFactor, const double leaky,const int extraParam)
{
    // Fixed Parameters
    int non_zero_units = ceil(Nr * Nr * density);
    // Wr declaration
    SparseMatrix<floatBO, 0x1> Wr(Nr, Nr);
    Wr.reserve(non_zero_units);
    Wr.setZero();
    Wr.data().squeeze();

    SparseMatrix<floatBO, 0x1> Eye(Nr, Nr);
    Eye.setIdentity();
    Eye = Eye * (1 - leaky);

    // Accumulate valid indexes
    std::unordered_set<std::pair<int, int>, pair_hash> valid_idx_hash;

    // Initialize Wr according to desired topology
    switch(_type)
    {
        // Random topology
        case 0: wr_random_fill(Wr, Nr, non_zero_units, valid_idx_hash);
                break;
        // Small World topology
        case 1: wr_swt_fill(Wr, Nr, extraParam, valid_idx_hash);
                break;
        // Cyclid Reservoir Jump topology
        case 2: wr_crj_fill(Wr, Nr, extraParam, valid_idx_hash);
                break;
    }
    // Perform spectral radius scaling
    if (_reservoirInizialization == "radius")
    {
        Wr = Wr * leaky + Eye;
        // Get scale radius for Echo State Property
        floatBO scale_radius = wr_scale_radius(Wr, scalingFactor);

        // Rescale Wr
        Wr = (Wr * scale_radius - Eye) / leaky;
    }
    else
    {
        // Rescaling according to Gallicchio 2019 paper (see documentation for details)
        floatBO scaling = 6/sqrt(density * Nr * 12) * scalingFactor;
        Wr = Wr * leaky + Eye;
        Wr = (Wr * scaling - Eye) / leaky;
    }

    // Quantization
    // Quantizer quant(8, false);
    // quant.quantize_matrix<SparseBO>(Wr);
    return Wr;
}

/**
 * @brief Initialize configuration structure according to optimization vector and parameters file
 * 
 * @param optParams Eigen Vector used by Limbo to define hyper-parameters
 * @param confParams YAML Node containing hyper-parameters at high level
 */
void EchoBay::Reservoir::init_LayerConfig(const Eigen::VectorXd &optParams, const YAML::Node confParams)
{
    // Reset layerConfig
    _layerConfig.clear();
    int nLayers = parse_config<int>("Nl", 0, 0, optParams, confParams, 1);
    int countScale = 1;
    int NrTemporary;
    // if (confParams["scaleIn"]["count"])
    // {
    //     countScale = confParams["scaleIn"]["count"].as<int>();
    // }
    std::string scaleInType = confParams["scaleIn"]["type"].as<std::string>();
    if (scaleInType == "dynamic")
    {
        if(confParams["scaleIn"]["count"]){
            // Check count minimum size
            countScale = confParams["scaleIn"]["count"].as<int>() >= 2 ? confParams["scaleIn"]["count"].as<int>() : 2;
        }else{
            countScale = 2;

        }
    }else{
        countScale = 1;
    }
    // Get reservoir scaling type
    if (confParams["rho"]["scaling"])
    {
        _reservoirInizialization = confParams["rho"]["scaling"].as<std::string>();
    }

    int scaleSize;
    for (int i = 0; i < nLayers; i++)
    {
        layerParameter layerTemp;
        NrTemporary = parse_config<int>("Nr", i, 0, optParams, confParams, 400);
        layerTemp.Nr = NrTemporary >= 2 ? NrTemporary : 2;
        layerTemp.density = parse_config<double>("density", i, 0, optParams, confParams, 1);
        layerTemp.desiredRho = parse_config<double>("rho", i, 0, optParams, confParams, 0.9);
        layerTemp.leaky = parse_config<double>("leaky", i, 0, optParams, confParams, 0.9);
        layerTemp.edgesJumps = parse_config<int>("edgesJumps", i, 0, optParams, confParams, 0);

        // scaleInput management
        scaleSize = (i == 0) ? _Nu : 1;
        layerTemp.scaleInput.resize(scaleSize);
        layerTemp.scaleInput[0] = parse_config<double>("scaleIn", i, 0, optParams, confParams, 1);

        for (int j = 1; (j < countScale) && (i == 0); j++)
        {
            layerTemp.scaleInput[j] = parse_config<double>("scaleIn", nLayers, j, optParams, confParams, 1);
        }
        for (int j = countScale; (j < _Nu) && (i == 0); j++)
        {
            layerTemp.scaleInput[j] = layerTemp.scaleInput[countScale - 1];
        }
        _layerConfig.push_back(layerTemp);
    }
     
    // Manage Small World topology
    if (_type == 1)
    {
        int elementsToCopy;
        ArrayI InIndex;
        ArrayI OutIndex;
        for (int i = 0; i < nLayers; ++i)
        {
            elementsToCopy = (int) floor(_layerConfig[i].Nr /5.0);
            InIndex.resize(elementsToCopy);
            OutIndex.resize(elementsToCopy);
            for (int j = 0; j < elementsToCopy; ++j)
            {
                InIndex[j] = j;
                OutIndex[j] = j + (int) floor(_layerConfig[i].Nr /2.0);
            }
            _WinIndex.push_back(InIndex);
            _WoutIndex.push_back(OutIndex);
        }
          
    }

}

/**
 * @brief Initialize configuration structure according to vector of key/value pairs
 * 
 * @param paramValue vector of stringdouble_t pairs with key and value
 */
void EchoBay::Reservoir::init_LayerConfig(std::vector<stringdouble_t> paramValue) // TO DO, if necessary, updated with different types
{
    _layerConfig.clear();
    int nLayers = paramValue[0]["Nl"](0);
    int countScale = paramValue[0]["scaleInCount"](0);
    int scaleSize;
    for (int i = 0; i < nLayers; i++)
    {
        layerParameter layerTemp;
        layerTemp.Nr = paramValue[i]["Nr"](0);
        layerTemp.density = paramValue[i]["density"](0);
        layerTemp.desiredRho = paramValue[i]["rho"](0);
        layerTemp.leaky = paramValue[i]["leaky"](0);
        // scaleInput management
        scaleSize = (i == 0) ? _Nu : 1;
        layerTemp.scaleInput.resize(scaleSize);
        layerTemp.scaleInput[0] = paramValue[i]["scaleIn"](0);
        for (int j = 1; (j < countScale) && (i == 0); j++)
        {
            layerTemp.scaleInput[j] = paramValue[i]["scaleIn"](j);
        }
        for (int j = countScale; (j < _Nu) && (i == 0); j++)
        {
            layerTemp.scaleInput[j] = layerTemp.scaleInput[countScale - 1];
        }
        _layerConfig.push_back(layerTemp);
    }
}


std::unordered_set<int> EchoBay::Reservoir::pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }   
    }
    return elems;
}

ArrayI EchoBay::Reservoir::pick(int Nr, int k)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    //int k = 2 * floor(Nr/5);
    std::unordered_set<int> elems = pickSet(Nr, k, gen);

    // ok, now we have a set of k elements. but now
    // it's in a [unknown] deterministic order.
    // so we have to shuffle it:

    std::vector<int> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    ArrayI output = Eigen::Map<ArrayI, Eigen::Unaligned>(result.data(), result.size()); 
    return output;
}

/**
 * @brief Getter function for Reservoir type
 * 
 * @return int internal Reservoir type
 */
int EchoBay::Reservoir::get_ReservoirType() const
{
    return _type;
}

/**
 * @brief Getter function for layerConfig as vector of layerParameter
 * 
 * @return std::vector<layerParameter> Layers configuration
 */
std::vector<layerParameter> EchoBay::Reservoir::get_LayerConfig() const
{
    return _layerConfig;
}

/**
 * @brief Getter function for SWT WinIndex
 * 
 * @return std::vector<int> Number of layers
 */
int EchoBay::Reservoir::get_nLayers() const
{
    return _nLayers;
}

/**
 * @brief Return the sum of Nr across layers
 * 
 * @param layer number of layers. Default -1 counts all layers
 * @return int sum of Nr across layers
 */
int EchoBay::Reservoir::get_fullNr(const int layer) const
{
    auto sumNr = [](int sum, const layerParameter& curr){return sum + curr.Nr;};
    auto finalLayer = (layer == -1) ? _layerConfig.end() : _layerConfig.begin() + layer;

    return 1 + std::accumulate(_layerConfig.begin(), finalLayer, 0, sumNr);
}

/**
 * @brief Getter function for SWT WinIndex
 * 
 * @return std::vector<std::vector<int>> Win Index
 */
std::vector<ArrayI> EchoBay::Reservoir::get_WinIndex() const
{
    return _WinIndex;
}

/**
 * @brief Getter function for SWT WoutIndex
 * 
 * @return std::vector<std::vector<int>> Wout Index
 */
std::vector<ArrayI> EchoBay::Reservoir::get_WoutIndex() const
{
    return _WoutIndex;
}

/**
 * @brief Return the sum of Nr across layers for SWT topology
 * 
 * @param layer number of layers. Default -1 counts all layers
 * @return int sum of SWT Nr across layers
 */
int EchoBay::Reservoir::get_NrSWT(const int layer) const
{
    auto sumNrSWT = [](int sum, const ArrayI curr){return sum + curr.size();};
    auto finalLayer = (layer == -1) ? _WoutIndex.end() : _WoutIndex.begin() + layer;

    return 1 + std::accumulate(_WoutIndex.begin(), finalLayer, 0, sumNrSWT);
}

/**
 * @brief Return memory occupation of the Reservoir object
 * 
 * @params confParams YAML Node containing hyper-parameters at high level
 * @return floatBO The sum along all the layers of the product between Nr and Density 
 */
floatBO EchoBay::Reservoir::return_net_dimension(const YAML::Node confParams) const
{
    int optNr = 0,optDensity = 0;
    floatBO NrUpper, NrLower, densityUpper, densityLower;

    if (confParams["Nr"]["type"].as<std::string>() == "dynamic")
    {
        optNr = 1;
        NrUpper = confParams["Nr"]["upper_bound"].as<int>();
        NrLower = confParams["Nr"]["lower_bound"].as<int>();
    }

    if (confParams["density"]["type"].as<std::string>() == "dynamic")
    {
        optDensity = 1;
        densityUpper = confParams["density"]["upper_bound"].as<floatBO>();
        densityLower = confParams["density"]["lower_bound"].as<floatBO>();
    }

    floatBO count;
    count = (optNr + optDensity) * _nLayers == 0 ? 1 : (optNr + optDensity) * _nLayers;
    floatBO memory = 0;
    for (int i = 0; i < _nLayers; ++i)
    {
        floatBO tempNr, tempDensity;
        if(optNr){
            tempNr = (_layerConfig[i].Nr - NrLower)/(NrUpper - NrLower);
            memory += tempNr;
        }
        if(optDensity){
            tempDensity =  (_layerConfig[i].density - densityLower)/(densityUpper - densityLower);
            memory += tempDensity;
        }
        
    }
    return memory/count;
}

/**
 * @brief Print optimizable parameters with a pretty table
 * 
 * @param nLayers Number of layers
 * @param nWashout Sample washout
 * @param lambda Ridge regression factor see also EchoBay::Wout_ridge(int rows, int cols, double lambda, Eigen::Ref<MatrixBO> biasedState, Eigen::Ref<MatrixBO> target)
 * for details
 */
void EchoBay::Reservoir::print_params(const int nLayers, const int nWashout, const double lambda)
{
    //std::cout << "\n";
#ifdef USE_TBB
    cout_mutex.lock();
#endif
    std::cout << "Nr" << std::string(14, ' ');

    // Switch topology
    switch(_type)
    {
        case 0: std::cout << "density" << std::string(9, ' ');
                break;
        case 1: std::cout << "edges" << std::string(11, ' ');
                break;
        case 2: std::cout << "jump" << std::string(12, ' ');
                break;
    }
    std::cout << "scaleInput" << std::string(8 * (_layerConfig[0].scaleInput.size() - 1) + 1, ' ')
              << "leaky" << std::string(11, ' ')
              << "rho" << std::endl;
    
    int active_units = 0;
    for (int i = 0; i < nLayers; i++)
    {
        std::cout << std::setprecision(5)
                  << std::fixed << _layerConfig[i].Nr << "\t\t";

        // Switch topology
        if(_type == 0)
        {
            std::cout << std::fixed << _layerConfig[i].density << "\t\t";
        }
        else
        {
            std::cout << std::fixed << _layerConfig[i].edgesJumps << "\t\t";
        }

        // Sum active_units ATTENTION the network must be configured before print_params
        active_units += WrLayers[i].nonZeros();

        for (size_t countScale = 0; countScale < _layerConfig[i].scaleInput.size(); countScale++)
        {
            std::cout << std::fixed << _layerConfig[i].scaleInput[countScale] << " ";
        }
        std::cout << std::string(8 * (i != 0) * (_layerConfig[0].scaleInput.size() - 1) + 3, ' ');
        std::cout << std::fixed << _layerConfig[i].leaky << std::string(9, ' ')
                  << std::fixed << _layerConfig[i].desiredRho
                  << std::endl;
    }
    // Print general values
    std::cout << "washout\t" << nWashout << std::endl;
    std::cout << "lambda\t" << lambda << std::endl;
    std::cout << "active units\t" << active_units << std::endl;
#ifdef USE_TBB
    cout_mutex.unlock();
#endif
}