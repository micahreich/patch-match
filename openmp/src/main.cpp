#include <stdint.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

using namespace std;

#define GET ()

int main() {
    // int width, height, chan;
    // unsigned char *rgb_image = stbi_load("src/image_0362.jpg", &width, &height, &chan, 3);

    // if (rgb_image == nullptr) {
    //     cerr << "Error loading image\n";
    //     return -1;
    // }

    // Image img(width, height, chan, rgb_image);
    // RGBPixel px = img(0, 0);

    // printf("Red value: %p\n", img.data);
    // printf("Red value: %d\n", px.r);
    // // cout << "Red value: " << pixel << endl;

    // stbi_image_free(rgb_image);
    return 0;
}