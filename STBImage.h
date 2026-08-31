#include <string>

struct STBImage {
    int width{0}, height{0}, channels{0};
    uint8_t *rgb_image{nullptr};
    //std::string filename{};

    bool loadImage(const std::string &name);
    void saveImage(const std::string &newName) const;
};