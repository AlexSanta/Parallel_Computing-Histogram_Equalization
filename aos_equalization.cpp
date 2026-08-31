#include "aos_equalization.h"

std::vector<Pixel> from_STB_to_AoS(const STBImage &stb){
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

std::array<int, 256> compute_histogram_Y(const std::vector<Pixel> &aos){
    std::array<int,256> hist{0};

    for(const auto& p: aos){
        float Y = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
        hist[(int)Y]++;
    }

    return hist;
}

void apply_equalization_AoS(std::vector<Pixel> &aos, std::array<std::uint8_t, 256> &equalized){
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

std::vector<uint8_t> AoS_to_STB_buffer(const std::vector<Pixel>& aos) {
    std::vector<uint8_t> buffer(aos.size() * 3);

    for (int i = 0; i < aos.size(); i++) {
        buffer[3*i] = aos[i].r;
        buffer[3*i + 1] = aos[i].g;
        buffer[3*i + 2] = aos[i].b;
    }

    return buffer;
}