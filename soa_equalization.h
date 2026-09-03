#include "STBImage.h"
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>

struct Image{
    std::vector<uint8_t> r;
    std::vector<uint8_t> g;
    std::vector<uint8_t> b;
};

Image from_STB_to_SoA(const STBImage& stb);

std::array<int, 256> compute_histogram_Y(const Image& soa);

void apply_equalization_AoS(Image& soa, std::array<std::uint8_t, 256>& equalized);

std::vector<uint8_t> AoS_to_STB_buffer(const Image& soa);

