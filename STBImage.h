#ifndef STBIMAGE_H
#define STBIMAGE_H

#include <string>
#include <cstdint>

struct STBImage {
    int width{0}, height{0}, channels{0};
    std::uint8_t *rgb_image{nullptr};

    ~STBImage();
    bool loadImage(const std::string &name);
    void saveImage(const std::string &newName) const;
};

#endif