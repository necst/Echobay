#include "include/IOUtils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>


int main(int argc, const char* argv[])
{
    MatrixBO input;
    std::string filename = argv[1];
    std::string output = argv[2]; 
    std::cout << filename << std::endl;
    read_matrix<MatrixBO>(filename, input);
    std::cout << input << std::endl;

    std::ofstream outfile;
    outfile.open(argv[2]);
    outfile << std::setprecision(16);
    for(int i=0; i< input.rows(); i++)
    {
        outfile << input.row(i) << std::endl;
    }
    outfile.close();
    return 0;
}