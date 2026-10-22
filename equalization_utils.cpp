#include <cmath>

#include "equalization_utils.h"

std::array<int, L> compute_cdf(const std::array<int, L> &hist){
    std::array<int, L> cdf{};

    int cumulative = 0;
    for(int i=0; i<L; i++) {
        cumulative += hist[i];
        cdf[i] = cumulative;
    }

    return cdf;
}

std::array<std::uint8_t, L> compute_equalized_LUT(const std::array<int, L> &cdf, int numPixels){
    std::array<std::uint8_t, L> equalized{};

    int cdf_min = 0;
    for(int i=0; i<L; i++){
        if(cdf[i]!=0){
            cdf_min = cdf[i];
            break;
        }
    }

    if (cdf_min == numPixels)
    {
        for(int i=0; i<L; i++)
            equalized[i] = i;
    }
    else
    {
        const int denom = numPixels - cdf_min;

        for(int i=0; i<L; i++){
            float val = (float)(cdf[i] - cdf_min) / denom;
            equalized[i] = (std::uint8_t)roundf(val * 255.0f);
        }
    }

    return equalized;

}