#ifndef COMPUTESTATE_HPP
#define COMPUTESTATE_HPP

#include "EchoBay.hpp"
#include "Reservoir.hpp"

namespace EchoBay
{ // Simple state ESN
void compute_state(Eigen::Ref<MatrixBO> dst, Eigen::Ref<MatrixBO> Win,
                  Eigen::Ref<SparseBO> Wr, const MatrixBO &src,
                  Eigen::Ref<ArrayBO> stateArr,
                  const Eigen::Ref<const ArrayI8> sampleState);
// Leaky state ESN
void compute_state(Eigen::Ref<MatrixBO> dst, Eigen::Ref<MatrixBO> Win,
                           Eigen::Ref<SparseBO> Wr, const MatrixBO &src,
                           Eigen::Ref<ArrayBO> stateArr, 
                           const Eigen::Ref<const ArrayI8> sampleState, floatBO leaky);
// Deep ESN
void compute_state(Eigen::Ref<MatrixBO> dst, const std::vector<MatrixBO> &WinL,
                  const std::vector<SparseBO> &WrL,
                  const MatrixBO &src, std::vector<ArrayBO> &stateMat,
                  const Eigen::Ref<const ArrayI8> sampleState,
                  const std::vector<layerParameter> &layerConfig);

// Single Deep ESN update
ArrayBO update_state(const std::vector<MatrixBO> &WinL,
                              const std::vector<SparseBO> &WrL,
                              const Eigen::Ref<ArrayBO> src, std::vector<ArrayBO> &stateMat,
                              const std::vector<layerParameter> &layerConfig);

void compute_output(const Eigen::Ref<MatrixBO> &Win, const Eigen::Ref<SparseBO> &Wr,
                    const Eigen::Ref<MatrixBO> &Wout, const std::vector<ArrayBO> input_data,
                    const Eigen::Ref<ArrayBO> start_state, const unsigned int Nu, const unsigned int n_outputs);
} // namespace EchoBay
#endif
