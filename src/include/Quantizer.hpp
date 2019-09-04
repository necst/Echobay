#ifndef QUANTIZER_HPP
#define QUANTIZER_HPP

#include "EigenConfig.hpp"

namespace EchoBay
{
    class Quantizer
    {
        public:
        Quantizer(const int bins, const bool uniform = true)
        {
            _bins = bins;
            _uniform = uniform;
            _deltaUni = (_bins%2 == 0) ? (2.0)/(double)(_bins-1) : (2.0)/(double)_bins;
        }
        template<typename T>
        void quantize_matrix(T &input)
        {
            if(_uniform)
            {
                input = input.unaryExpr(quantize_matrix_uni<floatBO>(_deltaUni));
            }
            else
            {
                input = input.unaryExpr(quantize_matrix_mu<floatBO>(_mu, _coeffMu, _deltaUni));
            }   
        }

        private:
        /*Uniform Quantization unary function*/
        template<typename Scalar>
        struct quantize_matrix_uni {
            quantize_matrix_uni(const double &deltaUni) : _deltaUni(deltaUni) {}
            const Scalar operator()(const Scalar& x) const
            {
                Scalar output = std::floor(x/_deltaUni)*_deltaUni + 0.5*_deltaUni; //std::round(x/_deltaUni)*_deltaUni;
                return output;
            }
            double _deltaUni; 
        };

        /*Non-Uniform Mu-Law Quantization unary function*/
        template<typename Scalar>
        struct quantize_matrix_mu {
            quantize_matrix_mu(const double &mu, const double &coeffMu, const double &deltaUni) 
                               : _mu(mu), _coeffMu(coeffMu), _deltaUni(deltaUni) {}
            const Scalar operator()(const Scalar& x) const
            {
                Scalar temp = _coeffMu * std::log(1 + _mu * std::fabs(x)) * std::copysign(1.0, x);
                Scalar output = std::floor(temp/_deltaUni)*_deltaUni + 0.5*_deltaUni; //std::round(temp/_deltaUni)*_deltaUni;
                return output;
            }
            double _mu, _coeffMu, _deltaUni; 
        };

        // Private variables
        int _bins;
        int _mu = 40; // Fixed compression
        // Mu-Law Quantization
        double _coeffMu = 1.0/std::log(1 + _mu); // Saves some calculations
        bool _uniform;
        double _deltaUni;
    };

}
#endif