#include <stdint.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image.h"

using namespace std;

#define GET ()

int main() {
    int width, height, chan;
    unsigned char *rgb_image = stbi_load("src/image_0362.jpg", &width, &height, &chan, 3);

    if (rgb_image == nullptr) {
        cerr << "Error loading image\n";
        return -1;
    }

    Image img(width, height, chan, rgb_image);
    unsigned char *pixel = img(0, 0);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char *pixel_img = img(i, j);
            unsigned char *pixel_og = rgb_image + (i * width + j) * chan;

            for (int k = 0; k < 3; k++)
                if (pixel_img[k] != pixel_og[k]) cerr << "Pixel Mismatch Error\n";
        }
    }

    printf("Red value: %p\n", img.data);
    printf("Red value: %d\n", pixel[0]);
    // cout << "Red value: " << pixel << endl;

    stbi_image_free(rgb_image);
}