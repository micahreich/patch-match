#ifndef  __UTILS_H__
#define  __UTILS_H__

// #ifndef PATCH_SIZE
// #error "PATCH_SIZE not found"
// #endif

// #ifndef HALF_SIZE
// #error "HALF_SIZE not found"
// #endif

#define MASK_UNOCCLUDED = 0;
#define MASK_OCCLUDED = !MASK_OCCLUDED;

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <random>

struct ImageSliceCoords {
    int row_start;
    int row_end;
    int col_start;
    int col_end;
};

const float GAUSSIAN_KERNEL[3][3] = {
    {0.075113608, 0.123841403, 0.075113608},
    {0.123841403, 0.204179956, 0.123841403},
    {0.075113608, 0.123841403, 0.075113608}
};

bool inBounds(int x, int y, int width, int height, int half_size=0) {
    return (x >= half_size && x < width - half_size && y >= half_size && y < height - half_size);
}

template<typename T>
struct Vec2 {
    T i, j;

    Vec2() : i(0), j(0) {}
    Vec2(T i, T j) : i(i), j(j) {}

    Vec2 operator+(const Vec2& other) const {
        return Vec2(i + other.i, j + other.j);
    }

    Vec2 operator-(const Vec2& other) const {
        return Vec2(i - other.i, j - other.j);
    }

    Vec2 operator*(const Vec2& other) const {
        return Vec2(i * other.i, j * other.j);
    }

    Vec2 operator/(const Vec2& other) const {
        return Vec2(i / other.i, j / other.j);
    }
};

typedef Vec2<size_t> Vec2l;
typedef Vec2<int> Vec2i;
typedef Vec2<float> Vec2f;

template<typename T>
struct Array2D {
    unsigned int height, width;
    T *data;

    Array2D() : height(0), width(0), data(nullptr) {}

    Array2D(unsigned int h, unsigned int w, T *init_data=nullptr) : height(h), width(w) {
        unsigned int n_items = height * width;
        data = new T[n_items];

        if (init_data) {
            memcpy(data, init_data, n_items * sizeof(T));
        }
    }

    Array2D(const Array2D& other) : height(other.height), width(other.width) {
        unsigned int n_items = height * width;
        data = new T[n_items];

        memcpy(data, other.data, n_items * sizeof(T));
    }

    ~Array2D() {
        delete[] data;
    }

    T& operator()(unsigned int r, unsigned int c) {
        return data[r * width + c];
    }

    const T& operator()(unsigned int r, unsigned int c) const {
        return data[r * width + c];
    }

    static Array2D<T> downsample(const Array2D<T>& array, int dx)
    {
        auto downsampled_height = array.height / dx, downsampled_width = array.width / dx;
        Array2D<T> downsampled_array(downsampled_height, downsampled_width);

        for (int r = 0; r < array.height; r += dx) {
            for (int c = 0; c < array.width; c += dx) {
                downsampled_array(r / dx, c / dx) = array(r, c);
            }
        }

        return downsampled_array;
    }

    static Array2D<T> pad(const Array2D<T>& array, unsigned int x_padding, unsigned int y_padding, bool constant_mode = false) {
        // Reallocate new space for image
        // Store padding values
        auto padded_height = array.height + 2*y_padding;
        auto padded_width = array.width + 2*x_padding;

        Array2D<T> padded_array(padded_height, padded_width);

        // Copy original array into the center of the padded array
        for (unsigned int r = 0; r < array.height; ++r) {
            for (unsigned int c = 0; c < array.width; ++c) {
                padded_array(r + y_padding, c + x_padding) = array(r, c);
            }
        }

        // Pad the top and bottom
        for (unsigned int r = 0; r < y_padding; ++r) {
            for (unsigned int c = 0; c < padded_width; ++c) {
                padded_array(r, c) = constant_mode ? T() : padded_array(y_padding, c);
                padded_array(r + y_padding + array.height, c) = constant_mode ? T() : padded_array(y_padding + array.height - 1, c);
            }
        }

        // Pad the left and right
        for (unsigned int r = 0; r < padded_height; ++r) {
            for (unsigned int c = 0; c < x_padding; ++c) {
                padded_array(r, c) = constant_mode ? T() : padded_array(r, x_padding);
                padded_array(r, c + x_padding + array.width) = constant_mode ? T() : padded_array(r, x_padding + array.width - 1);
            }
        }

        return padded_array;
    }

};

struct MaskStruct {
    unsigned int height, width;
    Array2D<bool> data;

    Mask() : height(0), width(0), data() {}

    Mask(unsigned int h, unsigned int w, Array2D<bool> other) : height(h), width(w), data(other) {}

    Mask(const Mask& other) : height(other.height), width(other.width), data(other.data) {}

    ~Mask() {
        
    }
};

struct GradientPair {
    float grad_x, grad_y;

    GradientPair() : grad_x(0.f), grad_y(0.f) {}
    GradientPair(float gx, float gy) : grad_x(gx), grad_y(gy) {}

    GradientPair operator+(const GradientPair& gp) const {
        return GradientPair(grad_x + gp.grad_x, grad_y + gp.grad_y);
    }

    GradientPair operator-(const GradientPair& gp) const {
        return GradientPair(grad_x - gp.grad_x, grad_y - gp.grad_y);
    }

    GradientPair operator*(const GradientPair& gp) const {
        return GradientPair(grad_x * gp.grad_x, grad_y * gp.grad_y);
    }

    GradientPair operator/(const GradientPair& gp) const {
        if(gp.grad_x == 0 || gp.grad_y == 0) {
            throw std::invalid_argument("Division by zero is not allowed.");
        }
        return GradientPair(grad_x / gp.grad_x, grad_y / gp.grad_y);
    }

    GradientPair& operator*=(const GradientPair& gp) {
        grad_x *= gp.grad_x;
        grad_y *= gp.grad_y;
        return *this;
    }
};

struct RGBPixel {
    unsigned char r, g, b;

    RGBPixel() : r(0), g(0), b(0) {}
    RGBPixel(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}

    RGBPixel operator+(const RGBPixel& p) const {
        return RGBPixel(r + p.r, g + p.g, b + p.b);
    }

    RGBPixel operator-(const RGBPixel& p) const {
        return RGBPixel(r - p.r, g - p.g, b - p.b);
    }

    RGBPixel operator*(const RGBPixel& p) const {
        return RGBPixel(r * p.r, g * p.g, b * p.b);
    }

    RGBPixel& operator*=(const RGBPixel& other) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    RGBPixel operator/(const RGBPixel& p) const {
        if(p.r == 0 || p.g == 0 || p.b == 0) {
            throw std::invalid_argument("Division by zero is not allowed.");
        }
        return RGBPixel(r / p.r, g / p.g, b / p.b);
    }

    RGBPixel operator*(float f) {
        return RGBPixel(static_cast<unsigned char>(r*f), static_cast<unsigned char>(g*f), static_cast<unsigned char>(b*f));
    }

};

// Function to generate Gaussian kernel
// std::vector<std::vector<double>> generateGaussianKernel(int kernelSize, double sigma) {
//     std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
//     double sum = 0.0;
//     int halfSize = kernelSize / 2;

//     for (int x = -halfSize; x <= halfSize; x++) {
//         for (int y = -halfSize; y <= halfSize; y++) {
//             kernel[x + halfSize][y + halfSize] = exp(-(x*x + y*y) / (2 * sigma*sigma)) / (2 * M_PI * sigma*sigma);
//             sum += kernel[x + halfSize][y + halfSize];
//         }
//     }

//     // Normalize the kernel
//     for (int i = 0; i < kernelSize; ++i)
//         for (int j = 0; j < kernelSize; ++j)
//             kernel[i][j] /= sum;

//     return kernel;
// }

// Function to apply Gaussian filter to an image
Array2D<RGBPixel> gaussianFilter(Array2D<RGBPixel>& image) {
    int height = image.height;
    int width = image.width;
    Array2D<RGBPixel> filtered_image(height, width);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            RGBPixel sum(0, 0, 0);

            // Iterate through structure block
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int px_r = r + dr, px_c = c + dc;
                    // TODO @dkrajews: fix the blurring around the edges
                    // see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html for info on different methods

                    // if (newX >= 0 && newX < height && newY >= 0 && newY < width) {
                    //     sum = (image(newX, newY) * GAUSSIAN_KERNEL[x + halfSize][y + halfSize]) + sum;
                    // }
                }
            }

            filtered_image(r, c) = sum;
        }
    }

    return filtered_image;
}

typedef Array2D<Vec2i> shift_map_t;
typedef Array2D<float> distance_map_t;
typedef Array2D<GradientPair> texture_t;
typedef Array2D<bool> mask_t;
typedef Array2D<RGBPixel> image_t;

// template<typename T>
// Array2D<T> downsampleArray(Array2D<T>& array, int dx) {
//     auto downsampled_height = array.height / dx, downsampled_width = array.width / dx;
//     Array2D<T> downsampled_array(downsampled_height, downsampled_width);

//     for (int r = 0; r < array.height; r += dx) {
//         for (int c = 0; c < array.width; c += dx) {
//             downsampled_array(r / dx, c / dx) = array(m, n);
//         }
//     }

//     return downsampled_array;
// }

mask_t structureBlockConvolve(const mask_t &mask, bool activeCenterVal,
                              const bool block[3][3], unsigned int half_size=0,
                              std::function<bool(bool, bool)> combineFn = [](bool a, bool b) { return a && b; })
{
    mask_t convolved_mask(mask); // Deep copy of the original mask

    for (int r = 0; r < mask.height; r++) {
        for (int c = 0; c < mask.width; c++) {
            if (mask(r, c) != activeCenterVal) continue;

            // Iterate through structure block
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int px_r = r + dr, px_c = c + dc;

                    if (!inBounds(px_c, px_r, mask.width, mask.height, half_size)) continue;
                    convolved_mask(px_r, px_c) = combineFn(convolved_mask(px_r, px_c), block[dr + 1][dc + 1]);
                }
            }
        }
    }

    return convolved_mask;
}

mask_t dilateMask(const mask_t &mask, unsigned int half_size=0) {
    const bool dilation_block[3][3] = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}
    };

    auto dilation_fn = [](bool a, bool b) { return a || b; };

    return structureBlockConvolve(mask, 1, dilation_block, half_size, dilation_fn);
}

mask_t erodeMask(const mask_t &mask, unsigned int half_size=0) {
    const bool erosion_block[3][3] = {
        {1, 0, 1},
        {0, 0, 0},
        {1, 0, 1}
    };

    auto erosion_fn = [](bool a, bool b) { return a && b; };

    return structureBlockConvolve(mask, 0, erosion_block, half_size, erosion_fn);
}

bool maskNotEmpty(const mask_t &mask) {
    for (int r = 0; r < mask.height; r++) {
        for (int c = 0; c < mask.width; c++) {
            if (mask(r, c)) return true;
        }
    }

    return false;
}

static int random_int(int lb, int ub) {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> dist(lb, ub);

    return dist(rng);
}


#endif