#ifndef SOA_EQUALIZATION_H
#define SOA_EQUALIZATION_H

#include <vector>
#include <array>
#include <cstdint>
#include <string>

#include "STBImage.h"
#include "equalization_utils.h"

struct Image{
    std::vector<std::uint8_t> r;
    std::vector<std::uint8_t> g;
    std::vector<std::uint8_t> b;
};

Image from_STB_to_SoA_sequential(const STBImage &stb);
Image from_STB_to_SoA_parallel(const STBImage &stb);

std::array<int, L> compute_histogram_Y_sequential(const Image &soa);
std::array<int, L> compute_histogram_Y_parallel(const Image &soa);

void apply_equalization_SoA_sequential(Image &soa, const std::array<std::uint8_t, L> &equalized);
void apply_equalization_SoA_parallel(Image &soa, const std::array<std::uint8_t, L> &equalized);

void update_STB_from_SoA_sequential(STBImage &stb, const Image &soa);
void update_STB_from_SoA_parallel(STBImage &stb, const Image &soa);

void process_SoA_image_sequential(const std::string& inputPath, const std::string& outputPath);
void process_SoA_image_parallel(const std::string& inputPath, const std::string& outputPath);

#endif
