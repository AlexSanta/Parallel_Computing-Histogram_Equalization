#include "equalization_utils.h"

std::array<int, 256> compute_cdf(const std::array<int, 256> &hist){
    std::array<int, 256> cdf{};

    int cumulative = 0;
    for(int i=0; i<256; i++) {
        cumulative += hist[i];
        cdf[i] = cumulative;
    }

    return cdf;
}

std::array<std::uint8_t, 256> compute_equalized_LUT(const std::array<int, 256> &cdf, int numPixels){
    std::array<std::uint8_t, 256> equalized{};

    int cdf_min = 0;
    for(int i=0; i<256; i++){
        if(cdf[i]!=0){
            cdf_min = cdf[i];
            break;
        }
    }

    for(int i=0; i<256; i++){
        float val = (float)(cdf[i] - cdf_min) / (numPixels - cdf_min);
        equalized[i] = (uint8_t)roundf(val * 255.0f);
    }

    return equalized;

}