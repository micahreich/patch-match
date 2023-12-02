#ifndef  __UTILS_H__
#define  __UTILS_H__

#ifndef PATCH_SIZE
#error "PATCH_SIZE not found"
#endif

#ifndef HALF_SIZE
#error "HALF_SIZE not found"
#endif

#include <stdio.h>
#include <string.h>

bool inBounds(int x, int y, int width, int height, bool outsidePadding = false) {
    int lower = outsidePadding ? HALF_SIZE : 0;
    return (x >= lower && x < width - lower && y >= lower && y < height - lower);
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

typedef Vec2<int> Vec2i;
typedef Vec2<float> Vec2f;

template<typename T>
struct ArrayND {
    unsigned int height, width;
    T *data;

    ArrayND(unsigned int h, unsigned int w, T *init_data=nullptr) : height(h), width(w) {
        unsigned int n_items = h * w;
        data = new T[n_itens];

        if (init_data) {
            memcpy(data, init_data, n_itens * sizeof(T));
        }
    }

    T& operator()(unsigned int r, unsigned int c) {
        return data[r * width + c];
    }

    ~ArrayND() {
        delete[] data;
    }
};

#endif