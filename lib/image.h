#ifndef  __IMAGE_H__
#define  __IMAGE_H__

#include <stdexcept>
#include <cstdio>

#include <stdint.h>
#include <string.h>

struct RGBPixel {
    unsigned char r, g, b;

    RGBPixel(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}
};

class Proxy {
public:
    Proxy(unsigned char* _data, int _channels, int _width = 0) : data(_data), channels(_channels), width(_width) {}

    Proxy operator[](int c) {
        return Proxy(data + c * channels, channels);
    }

    void operator=(const RGBPixel& pixel) {
        data[0] = pixel.r;
        data[1] = pixel.g;
        data[2] = pixel.b;
    }

    operator RGBPixel() const {
        return RGBPixel(data[0], data[1], data[2]);
    }

private:
    unsigned char* data;
    int channels;
    int width;
};

struct RGBImage {
    int width, height, channels;
    unsigned char* data;

    RGBImage(int w, int h, int c, unsigned char* data);
    ~RGBImage();

    Proxy operator[](int r) {
        return Proxy(data + r * width * channels, channels, width);
    }

    void setPatch(const int center_r, const int center_c, const int patch_size, const unsigned char* patch);
};

struct MonoImage {
    int width, height;
    unsigned char* data;

    MonoImage(RGBImage &img);
    ~MonoImage();

    Proxy operator[](int r) {
        return Proxy(data + r * width * 1, 1, width);
    }

    unsigned char operator()(int r, int c) const;
    void set(int r, int c, unsigned char pixel);
    void setPatch(const int center_r, const int center_c, const int patch_size, const unsigned char* patch);
};

// struct MonoImage {
//     int width, height;
//     unsigned char* data;

//     MonoImage(RGBImage img) {
//         this->width = img.width;
//         this->height = img.height;
//         this->data = nullptr;

//         const float R_ADJUSTMENT_COEFF = 0.2126;
//         const float G_ADJUSTMENT_COEFF = 0.7152;
//         const float B_ADJUSTMENT_COEFF = 0.0722;

//         if (img.data != nullptr) {
//             this->data = new unsigned char[width * height];
//             memset(this->data, 0, width * height * sizeof(unsigned char));

//             for (int r = 0; r < height; r++) {
//                 for (int c = 0; c < width; c++) {
//                     RGBPixel px = img(r, c);
//                     this->data[r * width + c] = static_cast<unsigned char>(
//                           R_ADJUSTMENT_COEFF * static_cast<float>(px.r)
//                         + G_ADJUSTMENT_COEFF * static_cast<float>(px.g)
//                         + B_ADJUSTMENT_COEFF * static_cast<float>(px.b)
//                     );
//                 }
//             }
//         }
//     }

//     ~MonoImage() {
//         delete[] data;
//     }
// };

#endif
