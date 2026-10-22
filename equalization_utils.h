#ifndef EQUALIZATION_UTILS_H
#define EQUALIZATION_UTILS_H

#include <array>
#include <cstdint>

constexpr int L = 256;

std::array<int,L> compute_cdf(const std::array<int,L>& hist);

std::array<std::uint8_t,L> compute_equalized_LUT(const std::array<int,L>& cdf, int numPixels);

#endif