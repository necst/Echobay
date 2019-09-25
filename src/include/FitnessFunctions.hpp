#ifndef FITNESSFUNCTIONS_HPP
#define FITNESSFUNCTIONS_HPP

#include <functional>
#include <stdexcept>
#if defined(USE_PYBIND)
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#endif
#ifdef USE_TBB
#include <tbb/mutex.h>
#endif
#include "EchoBay.hpp"

namespace EchoBay   
{
    /** Map different topologies of Reservoir */
    static std::map<std::string, int> problemTypes = {{"Regression", 0}, 
                                                  {"Classification", 1},
                                                  {"MemoryCapacity", 3},
                                                  {"External", 4}};

    /**
     * Comparator class to calculate various fitness functions
     * 
     * Other than various label transformations, a Comparator object provides a common
     * API (see EchoBay::Comparator::get_fitness(Eigen::Ref<MatrixBO> predict)) to template 
     * the calls to fitness functions.
     * 
     */
    class Comparator
    {
        public:
        Comparator(const std::string &problemType, const std::string &fitnessRule);
        ~Comparator() {};
        void set_label_size(int rows, int cols);
        floatBO get_fitness(Eigen::Ref<MatrixBO> predict);
        MatrixBO get_outputlabel() { return _outputLabel; };
        void set_targetLabel(     const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayI8> sampleState);
        MatrixBO get_targetMatrix(const Eigen::Ref<const MatrixBO> label, const Eigen::Ref<const ArrayI8> sampleState);

        void one_hot_encoding(Eigen::Ref<MatrixBO> Dst, MatrixBO Src);
        int get_nClasses();

        protected:
        static floatBO find_median(std::vector<floatBO> len);
        static floatBO find_median(Eigen::Ref<ArrayBO> len);
        //floatBO calc_error(std::vector<floatBO> predict, std::vector<floatBO> actual);

        static MatrixBO argmax_label(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual);
        void multi_out_encoding(Eigen::Ref<MatrixBO> input_data, Eigen::Ref<MatrixBO> target_data);
        static void ConfusionMatrix(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<Eigen::MatrixXi> ConfusionMat);

        static floatBO MSA(           Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
        static floatBO NRMSE(         Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
        static floatBO F1Mean(        Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
        static floatBO Accuracy(      Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
        static floatBO MemoryCapacity(Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
#if defined(USE_PYBIND)
        static floatBO ExtFitness(    Eigen::Ref<MatrixBO> predict, Eigen::Ref<MatrixBO> actual, int nOutput, Eigen::Ref<MatrixBO> OutputLabel, const std::string rule);
#endif
        int get_nClasses(const Eigen::Ref<const MatrixBO> label);   

        private:
        std::string _problemType = "Regression";
        std::string _fitnessRule = "NRMSE";
        int _rows, _cols, _nOutput, _outCols;
        int _nClasses = 1;
        MatrixBO _outputLabel;
        MatrixBO _targetMatrix;
        MatrixBO _targetLabel;
        std::function<floatBO(Eigen::Ref<MatrixBO>, Eigen::Ref<MatrixBO>, int, Eigen::Ref<MatrixBO>, const std::string rule)> fitnessFunction;
    };
} // namespace EchoBay

#endif
