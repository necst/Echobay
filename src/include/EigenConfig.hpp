#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <unordered_set>

/** floatBO defines the floating point precision to be used in EchoBay */
typedef double floatBO;
typedef Eigen::ArrayXd ArrayBO;
typedef Eigen::ArrayXi ArrayI;
typedef Eigen::Array<int8_t, Eigen::Dynamic, 1> ArrayI8;
typedef Eigen::MatrixXd MatrixBO;
typedef Eigen::VectorXcd VComplexBO;
typedef Eigen::Triplet<floatBO> TripletBO;
typedef Eigen::SparseMatrix<floatBO, 0x1> SparseBO;

// Structures for Hash Table and WR Filling
struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const
    {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};
typedef std::unordered_set<std::pair<int,int>, pair_hash >::iterator UOMIterator;
#endif