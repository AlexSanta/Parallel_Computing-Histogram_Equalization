#include <vector>
#include <array>
#include <cstdint>
#include "STBImage.h"

struct Pixel {
    std::uint8_t r, g, b;
};

std::vector<Pixel> from_STB_to_AoS(const STBImage& stb);

std::array<int, 256> compute_histogram_Y(const std::vector<Pixel>& aos);

void apply_equalization_AoS(std::vector<Pixel>& aos, std::array<std::uint8_t, 256>& equalized);

std::vector<uint8_t> AoS_to_STB_buffer(const std::vector<Pixel>& aos);
