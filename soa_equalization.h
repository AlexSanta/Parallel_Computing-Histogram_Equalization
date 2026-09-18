#include "STBImage.h"
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <omp.h>

struct Image{
    std::vector<uint8_t> r;
    std::vector<uint8_t> g;
    std::vector<uint8_t> b;
};

Image from_STB_to_SoA_sequential(const STBImage& stb);
Image from_STB_to_SoA_parallel(const STBImage& stb);

std::array<int, 256> compute_histogram_Y_sequential(const Image& soa);
std::array<int, 256> compute_histogram_Y_parallel(const Image& soa);

void apply_equalization_AoS_sequential(Image& soa, std::array<std::uint8_t, 256>& equalized);
void apply_equalization_AoS_parallel(Image& soa, std::array<std::uint8_t, 256>& equalized);

std::vector<uint8_t> AoS_to_STB_buffer_sequential(const Image& soa);
std::vector<uint8_t> AoS_to_STB_buffer_parallel(const Image& soa);

