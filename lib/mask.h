#ifndef  __MASK_H__
#define  __MASK_H__

#include <stdexcept>
#include <cstdio>

#include <stdint.h>
#include <string.h>

#include <image.h>

#define FREE true
#define MASKED !FREE

struct Mask {
    int width, height;
    bool *data;

    Mask(Image im) {
        this->width = im.width;
        this->height = im.height;
        this->data = nullptr;

        if (im.data != nullptr) {
            this->data = new bool[width * height * sizeof(bool)];
            memset(this->data, FREE, width * height * sizeof(bool));
        }
    }

    Mask(int w, int h, bool *data) : width(w), height(h) {
        this->data = nullptr;

        if (data != nullptr) {
            this->data = new bool[w * h * sizeof(bool)];
            memcpy(this->data, data, w * h * sizeof(bool));
        }
    }

    ~Mask() {
        delete[] data;
    }

    bool* operator()(int r, int c) const {
        #ifdef DEBUG
            if (!data) {
                throw std::runtime_error("Mask data is null");
            }

            if (r < 0 || r >= height || c < 0 || c >= width) {
                char msg[128];
                snprintf(msg, sizeof(msg), "Mask (%d, %d) is out of range for Mask of shape (%d, %d)", r, c, height, width);
                throw std::out_of_range(msg);
            }
        #endif

        return data + (r * width + c);
    }
};

#endif