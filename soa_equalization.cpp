#include "soa_equalization.h"

Image from_STB_to_SoA(const STBImage &stb){
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

std::array<int, 256> compute_histogram_Y(const Image &soa){
    std::array<int,256> hist{0};
    int numPixels = soa.r.size();

    for(int i=0; i<numPixels; i++){
        float Y = 0.299f * soa.r[i] + 0.587f * soa.g[i] + 0.114f * soa.b[i];
        hist[(int)Y]++;
    }

    return hist;
}

void apply_equalization_SoA(Image &soa, std::array<std::uint8_t, 256> &equalized){
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

std::vector<uint8_t> SoA_to_STB_buffer(const Image& soa) {
    int numPixels = soa.r.size();
    std::vector<uint8_t> buffer(numPixels * 3);

    for (int i = 0; i < numPixels; i++) {
        buffer[3*i] = soa.r[i];
        buffer[3*i + 1] = soa.g[i];
        buffer[3*i + 2] = soa.b[i];
    }

    return buffer;
}
