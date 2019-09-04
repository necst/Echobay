#ifndef TRAINREADOUT_HPP
#define TRAINREADOUT_HPP

#include "ComputeState.hpp"


namespace EchoBay
{ // Naive Wout_ridge
MatrixBO Wout_ridge(int rows, int cols, double lambda, Eigen::Ref<MatrixBO> biasedState, Eigen::Ref<MatrixBO> target);

// Wout through progressive Summation
MatrixBO readout_train(Reservoir &ESN, const MatrixBO &trainData,
                                const Eigen::Ref<const ArrayBO> sampleState,
                                Eigen::Ref<MatrixBO> target, double lambda, int blockStep);
// Compute Predicton
MatrixBO readout_predict(Reservoir &ESN, const MatrixBO &trainData, 
                                  const Eigen::Ref<const ArrayBO> sampleState,
                                  const Eigen::Ref<MatrixBO> Wout, int blockStep);

ArrayBO readout_predict(const Eigen::Ref<MatrixBO> Wout, const Eigen::Ref<const ArrayBO> ESNState);

// Kahan summation for avoiding loss of precision
void kahan_sum(MatrixBO Input, Eigen::Ref<MatrixBO> Sum, Eigen::Ref<MatrixBO> C);

MatrixBO clean_input(const Eigen::Ref<const MatrixBO> input, const Eigen::Ref<const ArrayBO> sampleState);

}
#endif