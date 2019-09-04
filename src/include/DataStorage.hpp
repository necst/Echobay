#ifndef DATASTORAGE_HPP
#define DATASTORAGE_HPP

#include "EigenConfig.hpp"
#include "Eigen/StdVector"
#include <numeric>
#include "IOUtils.hpp"

namespace EchoBay
{
    /**
     * Data management class
     * 
     * DataStorage objects embeds all input data, labels and sampling vectors. It
     * also includes data movement methods
     * 
     */
    class DataStorage
    {
        public:
        DataStorage() {};
        void load_data(const std::string dataFile, const std::string labelFile, const std::string type);
        void copy_data(Eigen::Ref<MatrixBO> data, Eigen::Ref<MatrixBO> label, const std::string type);
        ArrayBO set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const std::string type);
        MatrixBO get_data(const std::string type, const std::string select);
        ArrayBO get_sampleArray(const std::string type);

        MatrixBO _trainData, _trainLabel;
        MatrixBO _evalData, _evalLabel;
        ArrayBO _trainSampling, _evalSampling;
    };
}

#endif