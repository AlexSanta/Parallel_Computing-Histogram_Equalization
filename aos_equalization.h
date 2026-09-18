#include <vector>
#include <array>
#include <cstdint>
#include <omp.h>
#include "STBImage.h"

struct Pixel {
    std::uint8_t r, g, b;
};

std::vector<Pixel> from_STB_to_AoS_sequential(const STBImage& stb);
std::vector<Pixel> from_STB_to_AoS_parallel(const STBImage& stb);

std::array<int, 256> compute_histogram_Y_sequential(const std::vector<Pixel>& aos);
std::array<int, 256> compute_histogram_Y_parallel(const std::vector<Pixel>& aos);

void apply_equalization_AoS_sequential(std::vector<Pixel>& aos, std::array<std::uint8_t, 256>& equalized);
void apply_equalization_AoS_parallel(std::vector<Pixel>& aos, std::array<std::uint8_t, 256>& equalized);

std::vector<uint8_t> AoS_to_STB_buffer_sequential(const std::vector<Pixel>& aos);
std::vector<uint8_t> AoS_to_STB_buffer_parallel(const std::vector<Pixel>& aos);
