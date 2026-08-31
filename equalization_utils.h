#include <array>
#include <cstdint>
#include <cmath>

std::array<int,256> compute_cdf(const std::array<int,256>& hist);

std::array<uint8_t,256> compute_equalized_LUT(const std::array<int,256>& cdf, int numPixels);
