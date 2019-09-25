#ifndef ECHOBAY_HPP
#define ECHOBAY_HPP

#include "EigenConfig.hpp"

// Useful defines to reduce code size TODO move this in its own header
namespace EchoBay
{
    // Distinguish train and validation without strings
    const uint8_t Train = 0;
    const uint8_t Valid = 1;
    const uint8_t Test = Valid;

    // Distinguish Data and Label
    const uint8_t selData = 0;
    const uint8_t selLabel = 1;
}

#endif