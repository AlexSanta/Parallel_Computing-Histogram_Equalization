#ifndef AOS_EQUALIZATION_H
#define AOS_EQUALIZATION_H

#include <vector>
#include <array>
#include <cstdint>
#include <string>

#include "STBImage.h"
#include "equalization_utils.h"

struct Pixel {
    std::uint8_t r, g, b;
};

std::vector<Pixel> from_STB_to_AoS_sequential(const STBImage &stb);
std::vector<Pixel> from_STB_to_AoS_parallel(const STBImage &stb);

std::array<int, L> compute_histogram_Y_sequential(const std::vector<Pixel> &aos);
std::array<int, L> compute_histogram_Y_parallel(const std::vector<Pixel> &aos);

void apply_equalization_AoS_sequential(std::vector<Pixel> &aos, const std::array<std::uint8_t, L> &equalized);
void apply_equalization_AoS_parallel(std::vector<Pixel> &aos, const std::array<std::uint8_t, L> &equalized);

void update_STB_from_AoS_sequential(STBImage &stb, const std::vector<Pixel> &aos);
void update_STB_from_AoS_parallel(STBImage &stb, const std::vector<Pixel> &aos);

void process_AoS_image_sequential(const std::string& inputPath, const std::string& outputPath);
void process_AoS_image_parallel(const std::string& inputPath, const std::string& outputPath);

#endif