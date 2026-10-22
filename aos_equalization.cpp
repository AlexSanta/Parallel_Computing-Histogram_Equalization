#include <iostream>
#include <algorithm>
#include <omp.h>

#include "aos_equalization.h"

std::vector<Pixel> from_STB_to_AoS_sequential(const STBImage &stb){
    int numPixels = stb.width * stb.height;
    std::vector<Pixel> aos(numPixels);
    for(int p=0; p<numPixels; p++) {
        int i = p*stb.channels;
        aos[p].r = stb.rgb_image[i];
        aos[p].g = stb.rgb_image[i + 1];
        aos[p].b = stb.rgb_image[i + 2];
    }
    return aos;
}

std::vector<Pixel> from_STB_to_AoS_parallel(const STBImage &stb){
    int numPixels = stb.width * stb.height;
    std::vector<Pixel> aos(numPixels);

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for (int p = 0; p < numPixels; p++) {
        int i = p * stb.channels;
        aos[p].r = stb.rgb_image[i];
        aos[p].g = stb.rgb_image[i + 1];
        aos[p].b = stb.rgb_image[i + 2];
    }
    return aos;
}

std::array<int, L> compute_histogram_Y_sequential(const std::vector<Pixel> &aos){
    std::array<int,L> hist{0};

    for(const auto& p: aos){
        float Y = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        hist[(int)Y]++;
    }

    return hist;
}

std::array<int, L> compute_histogram_Y_parallel(const std::vector<Pixel> &aos){
    std::array<int,L> hist{0};
    int numPixels = aos.size();

    int numThreads = omp_get_max_threads();
    std::vector<std::array<int, L>> localHists(numThreads);

    #pragma omp parallel // NOLINT(openmp-use-default-none)
    {
        int tid = omp_get_thread_num();
        auto& lh = localHists[tid];
        lh.fill(0);

        #pragma omp for
        for(int i=0; i<numPixels; i++){
            float Y = 0.299f * aos[i].r + 0.587f * aos[i].g + 0.114f * aos[i].b;
            lh[(int) Y]++;
        }
    }

    for(int t=0; t<numThreads; t++)
        for(int i=0; i<L; i++)
            hist[i] += localHists[t][i];

    return hist;
}

void apply_equalization_AoS_sequential(std::vector<Pixel> &aos, const std::array<std::uint8_t, L> &equalized){
    for(auto& p: aos){
        float Y  = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        float Cb = -0.168736f * p.r - 0.331264f * p.g + 0.5f * p.b + 128;
        float Cr =  0.5f * p.r - 0.418688f * p.g - 0.081312f * p.b + 128;

        uint8_t Yeq = equalized[(int)Y];

        p.r = std::min(255, std::max((int)(Yeq + 1.402f * (Cr - 128)), 0));
        p.g = std::min(255, std::max((int)(Yeq - 0.344136f * (Cb - 128) - 0.714136f * (Cr - 128)), 0));
        p.b = std::min(255, std::max((int)(Yeq + 1.772f * (Cb - 128)), 0));
    }
}

void apply_equalization_AoS_parallel(std::vector<Pixel> &aos, const std::array<std::uint8_t, L> &equalized){
    int numPixels = aos.size();

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int i=0; i<numPixels; i++){
        float Y  = 0.299f * aos[i].r + 0.587f * aos[i].g + 0.114f * aos[i].b;
        float Cb = -0.168736f * aos[i].r - 0.331264f * aos[i].g + 0.5f * aos[i].b + 128;
        float Cr =  0.5f * aos[i].r - 0.418688f * aos[i].g - 0.081312f * aos[i].b + 128;

        uint8_t Yeq = equalized[(int)Y];

        aos[i].r = std::min(255, std::max((int)(Yeq + 1.402f * (Cr - 128)), 0));
        aos[i].g = std::min(255, std::max((int)(Yeq - 0.344136f * (Cb - 128) - 0.714136f * (Cr - 128)), 0));
        aos[i].b = std::min(255, std::max((int)(Yeq + 1.772f * (Cb - 128)), 0));
    }
}

void update_STB_from_AoS_sequential(STBImage &stb, const std::vector<Pixel> &aos) {
    int numPixels = stb.width * stb.height;

    for(int p = 0; p < numPixels; p++) {
        int i = p * stb.channels;
        stb.rgb_image[i] = aos[p].r;
        stb.rgb_image[i + 1] = aos[p].g;
        stb.rgb_image[i + 2] = aos[p].b;
    }
}

void update_STB_from_AoS_parallel(STBImage &stb, const std::vector<Pixel> &aos) {
    int numPixels = stb.width * stb.height;

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int p = 0; p < numPixels; p++) {
        int i = p * stb.channels;
        stb.rgb_image[i] = aos[p].r;
        stb.rgb_image[i + 1] = aos[p].g;
        stb.rgb_image[i + 2] = aos[p].b;
    }
}

void process_AoS_image_sequential(const std::string& inputPath, const std::string& outputPath){
    STBImage img;
    bool isRead = img.loadImage(inputPath);
    if (!isRead) {
        std::cerr << "ERROR: Failed to load image: " << inputPath << std::endl;
        return;
    }
    std::cout << "Image "<< inputPath << " loaded for AoS sequential equalization" << std::endl;

    double startTime = omp_get_wtime();

    auto aos = from_STB_to_AoS_sequential(img);
    auto hist = compute_histogram_Y_sequential(aos);
    auto cdf = compute_cdf(hist);
    auto equalized = compute_equalized_LUT(cdf, aos.size());
    apply_equalization_AoS_sequential(aos,equalized);
    update_STB_from_AoS_sequential(img, aos);

    double endTime = omp_get_wtime();
    std::cout << "Image processing time: " << endTime - startTime << std::endl;

    img.saveImage(outputPath);
    std::cout << "Equalized image saved in " << outputPath << std::endl;
}

void process_AoS_image_parallel(const std::string& inputPath, const std::string& outputPath){
    STBImage img;
    bool isRead = img.loadImage(inputPath);
    if (!isRead) {
        std::cerr << "ERROR: Failed to load image: " << inputPath << std::endl;
        return;
    }
    std::cout << "Image "<< inputPath << " loaded for AoS parallel equalization with "
              << omp_get_max_threads() << " threads" <<std::endl;

    double startTime = omp_get_wtime();

    auto aos = from_STB_to_AoS_parallel(img);
    auto input_hist = compute_histogram_Y_parallel(aos);
    auto cdf = compute_cdf(input_hist);
    auto equalized = compute_equalized_LUT(cdf, aos.size());
    apply_equalization_AoS_parallel(aos,equalized);
    update_STB_from_AoS_parallel(img, aos);

    double endTime = omp_get_wtime();
    std::cout << "Image processing time: " << endTime - startTime << std::endl;

    img.saveImage(outputPath);
    std::cout << "Equalized image saved in " << outputPath << std::endl;
}