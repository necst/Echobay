#ifndef DATASTORAGE_HPP
#define DATASTORAGE_HPP

#include "EchoBay.hpp"
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
        void load_data(const std::string dataFile, const std::string labelFile, const uint8_t type);
        void copy_data(Eigen::Ref<MatrixBO> data, Eigen::Ref<MatrixBO> label, const uint8_t type);
        ArrayI8 set_sampleArray(Eigen::Ref<MatrixBO> samplingData, int nWashout, bool init_flag, const std::string &problemType, const uint8_t type);
        MatrixBO get_data(const uint8_t type, const uint8_t select) const;
        int get_dataCols(const uint8_t type) const;
        int get_dataLength(const uint8_t type) const;
        ArrayI8 get_sampleArray(const uint8_t type) const;
        std::vector<ArrayI8> get_samplingBatches(const uint8_t type) const;
        int get_dataOffset(const uint8_t type, const int batch) const;
        int get_maxSamples(const uint8_t type, const int batch = -1) const;

        private:
        int get_nBatches(Eigen::Ref<MatrixBO> samplingData);
        std::array<std::vector<ArrayI8>, 2> _samplingBatches;
        std::array<MatrixBO, 2> _seriesData;
        std::array<MatrixBO, 2> _seriesLabel;
        std::array<ArrayI8, 2> _samplingFull;
        std::array<std::vector<int>, 2> _dataOffset;
    };
}

#endif