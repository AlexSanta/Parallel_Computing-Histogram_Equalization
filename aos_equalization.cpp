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

std::array<int, 256> compute_histogram_Y_sequential(const std::vector<Pixel> &aos){
    std::array<int,256> hist{0};

    for(const auto& p: aos){
        float Y = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        hist[(int)Y]++;
    }

    return hist;
}

std::array<int, 256> compute_histogram_Y_parallel(const std::vector<Pixel> &aos){
    std::array<int,256> hist{0};
    int numPixels = aos.size();

    int numThreads = omp_get_max_threads();
    std::vector<std::array<int, 256>> localHists(numThreads);

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
        for(int i=0; i<256; i++)
            hist[i] += localHists[t][i];

    return hist;
}

void apply_equalization_AoS_sequential(std::vector<Pixel> &aos, std::array<std::uint8_t, 256> &equalized){
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

void apply_equalization_AoS_parallel(std::vector<Pixel> &aos, std::array<std::uint8_t, 256> &equalized){
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

std::vector<uint8_t> AoS_to_STB_buffer_sequential(const std::vector<Pixel>& aos) {
    std::vector<uint8_t> buffer(aos.size() * 3);

    for(int i = 0; i < aos.size(); i++) {
        buffer[3*i] = aos[i].r;
        buffer[3*i + 1] = aos[i].g;
        buffer[3*i + 2] = aos[i].b;
    }

    return buffer;
}

std::vector<uint8_t> AoS_to_STB_buffer_parallel(const std::vector<Pixel>& aos) {
    std::vector<uint8_t> buffer(aos.size() * 3);
    int numPixels = aos.size();

    #pragma omp parallel for // NOLINT(openmp-use-default-none)
    for(int i = 0; i < numPixels; i++) {
        buffer[3*i] = aos[i].r;
        buffer[3*i + 1] = aos[i].g;
        buffer[3*i + 2] = aos[i].b;
    }

    return buffer;
}