#include "IOUtils.hpp" // TBB Global variable

#if !defined(ESP_PLATFORM)
/**
 * @brief Replace a {{x}} tag with a number in a string
 * 
 * @param inputString Input string with or without {{x}} tag
 * @param number Number to be applied
 * @return std::string Output string
 */
std::string replace_tag(const std::string &inputString, int number)
{
    std::string outputString;
    outputString = inputString;
    boost::replace_all(outputString, "{{x}}", std::to_string(number));
    return outputString;
}

/**
 * @brief Load a Eigen Matrix from a csv file
 * 
 * @param filename Path to the input file
 * @param output Eigen Matrix with loaded data
 */
void load_csv(std::string filename, MatrixBO &output)
{
    std::string header;
    char *line;
    char *tokens;
    size_t cols;
    size_t colIdx;
    size_t maxRows;

    // Load input from file
    io::LineReader inputData(filename);
    header = std::string(inputData.next_line());
    cols = std::count(header.begin(), header.end(), ',') + 1;

    // Resize matrix before filling it
    output.conservativeResize(1, cols);
    maxRows = 0;
    while ((line = inputData.next_line()))
    {
        ++maxRows;
    }
    output.conservativeResize(maxRows, cols);

    // Reset file
    inputData.reset_file();
    // read first line again
    inputData.next_line();

    // Read data
    for (size_t row = 0; row < maxRows; ++row)
    {
        line = inputData.next_line();
        tokens = strtok(line, ",");
        for (colIdx = 0; colIdx < cols; colIdx++)
        {
            output(row, colIdx) = atof(tokens);
            tokens = strtok(NULL, ",");
        }
    }
}
#endif

/**
 * @brief Save current configuration on a portable YAML file
 * 
 * @param filename Path to the file
 * @param YAML_CONF Current configuration strucutred as YAML Node
 * @param x Limbo configuration vector of hyper-parameters
 */
void save_config(const std::string &filename, YAML::Node YAML_CONF, const Eigen::VectorXd x)
{
    int idx;
    
    std::string variableName;    
    std::ofstream outputFile;

    //Output destination as configuration param??
    outputFile.open(filename, std::ios::out);
    YAML::Emitter output_emit;
    
    YAML::Node confCopy;
    confCopy = YAML_CONF;
    output_emit << YAML::BeginDoc;

    int nLayers = confCopy["Nl"]["value"].as<int>();
    int nDofLayer = confCopy["input_dimension_layer"].as<int>(); // Number of optimizable parameters per layer
    int nDofOffset = confCopy["input_dimension_general"].as<int>(); // Number of optimizable parameters general

    int scaleInCount = 1;// = confCopy["scaleIn"]["count"].as<int>();
    std::string scaleInType = confCopy["scaleIn"]["type"].as<std::string>();
    if (scaleInType == "dynamic")
    {
        if(confCopy["scaleIn"]["count"]){
            // Check count minimum size
            scaleInCount = confCopy["scaleIn"]["count"].as<int>() >= 2 ? confCopy["scaleIn"]["count"].as<int>() : 2;
        }else{
            scaleInCount = 1;
        }
    }

    std::vector<int> indices;
    std::vector<std::string> names;
    
    for (auto it = tMapGeneral.cbegin(); it != tMapGeneral.cend(); ++it)
    {  
        variableName = it->first;
        if (confCopy[variableName])
        {
            if (confCopy[variableName]["type"].as<std::string>() == "dynamic")
            {
                indices.push_back(confCopy[variableName]["index"].as<int>());
                names.push_back(variableName);
            }
        }
    }

    std::string layerNumber;
    for (int i = 0; i < nLayers; i++)
    {   
        layerNumber = "L" +std::to_string(i) ;

        for (auto it = tMapLayer.cbegin(); it != tMapLayer.cend(); ++it)
        {  
            variableName = it->first;
            if (confCopy[variableName])
            {
                if (confCopy[variableName]["type"].as<std::string>() == "dynamic")
                {

                    idx = confCopy[variableName]["index"].as<int>();
                    indices.push_back(idx + i*nDofLayer + nDofOffset);
                    names.push_back(variableName+layerNumber);
                }
            }
        }
    }
    
    int maxOffset = nLayers * nDofLayer + nDofOffset;
    if (scaleInCount > 1)
    {
        for (int i = maxOffset; i < maxOffset + scaleInCount-1; ++i)
        {
            indices.push_back(i);
            names.push_back("extraScaleIn");
        }
    }


    for (int i = 0; i < x.size(); ++i)
    {
        std::stringstream roundedValue;

        roundedValue << std::setprecision(5)
                     << std::fixed << x(i); 
        confCopy["x"].push_back(roundedValue.str());
    }
    confCopy["x"].SetStyle(YAML::EmitterStyle::Flow);

    std::string comment;

    for (auto i: sort_indexes(indices)) {
      comment += names[i] + " ";
    }

    output_emit << confCopy;
    output_emit << YAML::Comment(comment);

    output_emit << YAML::EndDoc;

    // Emit file
    outputFile << output_emit.c_str();
    outputFile.close();
}

/**
 * @brief Concatenate two Eigen Matrices
 * 
 * @param a Matrix a
 * @param b Matrix b
 * @return MatrixBO Concatenated matrices
 */
MatrixBO cat_matrix(const MatrixBO &a, const MatrixBO &b)
{
    MatrixBO output;
    output.resize(a.rows() + b.rows(), a.cols());
    output.block(0, 0, a.rows(), a.cols()) = a;
    output.block(a.rows(), 0, b.rows(), b.cols()) = b;
    return output;
}

/**
 * @brief Concatenate two Eigen Arrays
 * 
 * @param a Array a
 * @param b Array b
 * @return ArrayBO Concatenated arrays
 */
ArrayBO cat_array(const ArrayBO &a, const ArrayBO &b)
{
    ArrayBO output;
    output.resize(a.size() + b.size());
    output.head(a.size()) = a;
    output.tail(b.size()) = b;
    return output;
}