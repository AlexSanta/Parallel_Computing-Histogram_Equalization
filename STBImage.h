#include <string>
#include <cstdint>
#include "stb_image.h"
#include "stb_image_write.h"

struct STBImage {
    int width{0}, height{0}, channels{0};
    std::uint8_t *rgb_image{nullptr};

    bool loadImage(const std::string &name);
    void saveImage(const std::string &newName) const;
};