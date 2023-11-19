#include <image.h>

RGBImage::RGBImage(int w, int h, int c, unsigned char* data) : width(w), height(h), channels(c) {
    this->data = nullptr;

    if (data != nullptr) {
        this->data = new unsigned char[w * h * c * sizeof(unsigned char)];
        memcpy(this->data, data, w * h * c * sizeof(unsigned char));
    }
}

RGBImage::~RGBImage() {
    delete[] data;
}