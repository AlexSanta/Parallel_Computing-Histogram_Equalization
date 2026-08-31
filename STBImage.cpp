#include "STBImage.h"
#include "stb_image.h"
#include "stb_image_write.h"

bool STBImage::loadImage(const std::string &name) {
    rgb_image = stbi_load(name.c_str(), &width, &height, &channels, 3);
    if (!rgb_image)
        return false;
    else {
        //filename = name;
        return true;
    }
}

void STBImage::saveImage(const std::string &newName) const {
    stbi_write_jpg(newName.c_str(), width, height, channels, rgb_image, width * channels);
}

