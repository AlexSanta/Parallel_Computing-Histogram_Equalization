#include <iostream>
#include <algorithm>
#include <omp.h>

#include "soa_equalization.h"

Image from_STB_to_SoA_sequential(const STBImage &stb){
    int numPixels = stb.width * stb.height;
    Image soa;
    soa.r.resize(numPixels);
    soa.g.resize(numPixels);
    soa.b.resize(numPixels);

    for(int p=0; p<numPixels; p++) {
        int i = p*stb.channels;
        soa.r[p] = stb.rgb_image[i];
        soa.g[p] = stb.rgb_image[i + 1];
        soa.b[p] = stb.rgb_image[i + 2];
    }

    return soa;
}

Image from_STB_to_SoA_parallel(const STBImage &stb){
    int numPixels = stb.width * stb.height;
    Image soa;
    soa.r.resize(numPixels);
    soa.g.resize(numPixels);
    soa.b.resize(numPixels);

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int p=0; p<numPixels; p++) {
        int i = p*stb.channels;
        soa.r[p] = stb.rgb_image[i];
        soa.g[p] = stb.rgb_image[i + 1];
        soa.b[p] = stb.rgb_image[i + 2];
    }
    return soa;
}

std::array<int, L> compute_histogram_Y_sequential(const Image &soa){
    std::array<int,L> hist{0};
    int numPixels = soa.r.size();

    for(int i=0; i<numPixels; i++){
        float Y = 0.299f * soa.r[i] + 0.587f * soa.g[i] + 0.114f * soa.b[i];
        hist[(int)Y]++;
    }

    return hist;
}

std::array<int, L> compute_histogram_Y_parallel(const Image &soa){
    std::array<int,L> hist{0};
    int numPixels = soa.r.size();

    int numThreads = omp_get_max_threads();
    std::vector<std::array<int, L>> localHists(numThreads);

    #pragma omp parallel // NOLINT(openmp-use-default-none)
    {
        int tid = omp_get_thread_num();
        auto& lh = localHists[tid];
        lh.fill(0);

        #pragma omp for
        for(int i = 0; i < numPixels; i++) {
            float Y = 0.299f * soa.r[i] + 0.587f * soa.g[i] + 0.114f * soa.b[i];
            lh[(int) Y]++;
        }
    }

    for(int t=0; t<numThreads; t++)
        for(int i=0; i<L; i++)
            hist[i]+=localHists[t][i];

    return hist;
}

void apply_equalization_SoA_sequential(Image &soa, const std::array<std::uint8_t, L> &equalized){
    int numPixels = soa.r.size();

    for(int i=0; i<numPixels; i++){
        float Y  = 0.299f * soa.r[i] + 0.587f * soa.g[i] + 0.114f * soa.b[i];
        float Cb = -0.168736f * soa.r[i] - 0.331264f * soa.g[i] + 0.5f * soa.b[i] + 128;
        float Cr =  0.5f * soa.r[i] - 0.418688f * soa.g[i] - 0.081312f * soa.b[i] + 128;

        uint8_t Yeq = equalized[(int)Y];

        soa.r[i] = std::min(255, std::max((int)(Yeq + 1.402f * (Cr - 128)), 0));
        soa.g[i] = std::min(255, std::max((int)(Yeq - 0.344136f * (Cb - 128) - 0.714136f * (Cr - 128)), 0));
        soa.b[i] = std::min(255, std::max((int)(Yeq + 1.772f * (Cb - 128)), 0));
    }
}

void apply_equalization_SoA_parallel(Image &soa, const std::array<std::uint8_t, L> &equalized){
    int numPixels = soa.r.size();

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int i=0; i<numPixels; i++){
        float Y  = 0.299f * soa.r[i] + 0.587f * soa.g[i] + 0.114f * soa.b[i];
        float Cb = -0.168736f * soa.r[i] - 0.331264f * soa.g[i] + 0.5f * soa.b[i] + 128;
        float Cr =  0.5f * soa.r[i] - 0.418688f * soa.g[i] - 0.081312f * soa.b[i] + 128;

        uint8_t Yeq = equalized[(int)Y];

        soa.r[i] = std::min(255, std::max((int)(Yeq + 1.402f * (Cr - 128)), 0));
        soa.g[i] = std::min(255, std::max((int)(Yeq - 0.344136f * (Cb - 128) - 0.714136f * (Cr - 128)), 0));
        soa.b[i] = std::min(255, std::max((int)(Yeq + 1.772f * (Cb - 128)), 0));
    }
}

void update_STB_from_SoA_sequential(STBImage &stb, const Image &soa) {
    int numPixels = soa.r.size();

    for(int p = 0; p < numPixels; p++) {
        int i = p * stb.channels;
        stb.rgb_image[i] = soa.r[p];
        stb.rgb_image[i + 1] = soa.g[p];
        stb.rgb_image[i + 2] = soa.b[p];
    }

}

void update_STB_from_SoA_parallel(STBImage &stb, const Image &soa) {
    int numPixels = soa.r.size();

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int p = 0; p < numPixels; p++) {
        int i = p * stb.channels;
        stb.rgb_image[i] = soa.r[p];
        stb.rgb_image[i + 1] = soa.g[p];
        stb.rgb_image[i + 2] = soa.b[p];
    }
}

void process_SoA_image_sequential(const std::string& inputPath, const std::string& outputPath){
    STBImage img;
    bool isRead = img.loadImage(inputPath);
    if (!isRead) {
        std::cerr << "ERROR: Failed to load image: " << inputPath << std::endl;
        return;
    }
    std::cout << "Image "<< inputPath << " loaded for SoA sequential equalization" << std::endl;

    double startTime = omp_get_wtime();

    auto soa = from_STB_to_SoA_sequential(img);
    auto hist = compute_histogram_Y_sequential(soa);
    auto cdf = compute_cdf(hist);
    auto equalized = compute_equalized_LUT(cdf, soa.r.size());
    apply_equalization_SoA_sequential(soa, equalized);
    update_STB_from_SoA_sequential(img, soa);

    double endTime = omp_get_wtime();
    std::cout << "Image processing time: " << endTime - startTime << std::endl;

    img.saveImage(outputPath);
    std::cout << "Equalized image saved in " << outputPath << std::endl;
}

void process_SoA_image_parallel(const std::string& inputPath, const std::string& outputPath){
    STBImage img;
    bool isRead = img.loadImage(inputPath);
    if (!isRead) {
        std::cerr << "ERROR: Failed to load image: " << inputPath << std::endl;
        return;
    }
    std::cout << "Image "<< inputPath << " loaded for SoA parallel equalization with "
              << omp_get_max_threads() << " threads" <<std::endl;

    double startTime = omp_get_wtime();

    auto soa = from_STB_to_SoA_parallel(img);
    auto hist = compute_histogram_Y_parallel(soa);
    auto cdf = compute_cdf(hist);
    auto equalized = compute_equalized_LUT(cdf, soa.r.size());
    apply_equalization_SoA_parallel(soa, equalized);
    update_STB_from_SoA_parallel(img, soa);

    double endTime = omp_get_wtime();
    std::cout << "Image processing time: " << endTime - startTime << std::endl;

    img.saveImage(outputPath);
    std::cout << "Equalized image saved in " << outputPath << std::endl;
}
