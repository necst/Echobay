#ifndef IOUTILS_HPP
#define IOUTILS_HPP

#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>

#if !defined(ESP_PLATFORM)
#define CSV_IO_NO_THREAD
#include "csv.h"
#endif

#include "yaml-cpp/yaml.h"
#include <boost/algorithm/string/replace.hpp>
#include "EchoBay.hpp"
#include <unsupported/Eigen/src/SparseExtra/MarketIO.h>

/**
 * @brief Write a Eigen Matrix to file in binary format
 * 
 * @tparam Matrix Class of Eigen Matrix to be written
 * @param filename Path to the file or filename
 * @param matrix Input matrix
 */
template <class Matrix>
void write_matrix(const std::string &filename, const Matrix &matrix)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
    out.write((char *)(&rows), sizeof(typename Matrix::Index));
    out.write((char *)(&cols), sizeof(typename Matrix::Index));
    out.write((char *)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
    out.close();
}

/**
 * @brief Read a Eigen Matrix from a file in binary format
 * 
 * @tparam Matrix Class of Eigen Matrix to be written
 * @param filename Path to the file or filename
 * @param matrix Output matrix
 */
template <class Matrix>
void read_matrix(const std::string &filename, Matrix &matrix)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open())
    {
        typename Matrix::Index rows = 0, cols = 0;
        in.read((char *)(&rows), sizeof(typename Matrix::Index));
        in.read((char *)(&cols), sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read((char *)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
        in.close();
    }
    else
    {
        std::cout << "Error opening " << std::string(filename) << std::endl;
    }
}

/**
 * @brief Write a Eigen Array to file in binary format
 * 
 * @tparam Array Class of Eigen Array to be written
 * @param filename Path to the file or filename
 * @param array Input Array
 */
template <class Array>
void write_array(const std::string &filename, const Array &array)
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Array::Index rows = array.rows();
    out.write((char *)(&rows), sizeof(typename Array::Index));
    out.write((char *)array.data(), rows * sizeof(typename Array::Scalar));
    out.close();
}

/**
 * @brief Read a Eigen Array from a file in binary format
 * 
 * @tparam Array Class of Eigen Array to be read
 * @param filename Path to the file or filename
 * @param array Output Array
 */
template <class Array>
void read_array(const std::string &filename, Array &array)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open())
    {
        typename Array::Index rows = 0;
        in.seekg(0, in.beg);
        in.read((char *)(&rows), sizeof(typename Array::Index));
        array.resize(rows);
        in.read((char *)array.data(), rows * sizeof(typename Array::Scalar));
        in.close();
    }
    else
    {
        std::cout << "Error opening " << std::string(filename) << std::endl;
    }
}

/**
 * @brief Concatenate two std::vector
 * 
 * @tparam T Template of the std::vector
 * @param a Vector a
 * @param b Vector b
 * @return std::vector<T> Concatenated vector 
 */
template <typename T>
std::vector<T> cat_vector(const std::vector<T> &a, const std::vector<T> &b)
{
    std::vector<T> output;
    output.reserve(a.size() + b.size());
    output.insert(output.end(), a.begin(), a.end());
    output.insert(output.end(), b.begin(), b.end());
    return output;
}

/**
 * @brief Sort the indexes of a std::vector
 * 
 * @tparam T Template of the std::vector
 * @param v Input vector
 * @return std::vector<size_t> Sorted indexes
 */
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

typedef std::pair<std::string, std::string> stringpair_t;
const std::vector<stringpair_t> tMapLayer{
    stringpair_t("Nr", "int"),
    stringpair_t("density", "double"),
    stringpair_t("scaleIn", "double"),
    stringpair_t("rho", "double"),
    stringpair_t("leaky", "double"),
    stringpair_t("edges", "int")};
    
const std::vector<stringpair_t> tMapGeneral{
    stringpair_t("Nl", "int"),
    stringpair_t("washout_sample", "int"),
    stringpair_t("lambda", "double")};

void save_config(const std::string &filename, YAML::Node YAML_CONF, const Eigen::VectorXd x);

#if !defined(ESP_PLATFORM)
std::string replace_tag(const std::string &input_string, int number);
void load_csv(std::string filenameData, MatrixBO &arrValue);
#endif

MatrixBO cat_matrix(const MatrixBO &a, const MatrixBO &b);
ArrayBO cat_array(const ArrayBO &a, const ArrayBO &b);

#endif
