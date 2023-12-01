#include <stdint.h>
#include <iostream>

#define PATCH_SIZE 7
#define HALF_SIZE PATCH_SIZE / 2
#define BRUSH_SIZE_MULT 0.2f

#define TOGGLE_FILL_ON '1'        // ASCII for 1
#define TOGGLE_ERASE_ON '2'       // ASCII for 2
#define TOGGLE_BRUSH_RAD_INCR 'x' // ASCII for x
#define TOGGLE_BRUSH_RAD_DECR 'z' // ASCII for z
#define TOGGLE_SAVE 's'

const unsigned char MASK_COLOR[] = {0, 255, 0};
const unsigned char COLOR_WHITE[] = {255,255,255};
const unsigned char COLOR_BLACK[] = {0,0,0};

#define cimg_use_png
#include "CImg.h"
#include "utils.h"

using namespace cimg_library;
using namespace std;

enum FillMode {
    ERASE = 1,
    FILL = 0
};


void maskFillPatch(CImg<unsigned char> &mask, CImg<unsigned char> &masked_image, CImg<unsigned char> &original_image,
                   int x, int y, FillMode curr_mode, int brush_radius) {
    for (int i = -brush_radius; i <= brush_radius; i++) {
        for (int j = -brush_radius; j <= brush_radius; j++) {
            if (i*i + j*j <= brush_radius*brush_radius) {
                int circle_x = x + j;
                int circle_y = y + i;

                if (inBounds(circle_x, circle_y, mask.width(), mask.height())) {
                    mask(circle_x, circle_y) = curr_mode;

                    for (int k = 0; k < masked_image.spectrum(); k++) {
                        if (curr_mode == ERASE) {
                            masked_image(circle_x, circle_y, 0, k) = original_image(circle_x, circle_y, 0, k);
                        } else if (curr_mode == FILL) {
                            masked_image(circle_x, circle_y, 0, k) = MASK_COLOR[k];
                        }
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    // Read in user image with -i command line flag, or use lena.png as default
    const char *image_path = "src/lena.png";
    int opt;
    while ((opt = getopt(argc, argv, "i:")) != -1) {
        switch (opt) {
            case 'i':
                image_path = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-i image_path]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    // Create the image and masked images to display
    CImg<unsigned char> image(image_path);
    CImg<unsigned char> masked_image(image);

    int width = image.width();
    int height = image.height();
    int channels = image.spectrum();
    int min_dimension = min(width, height);

    CImg<unsigned char> mask(width, height, 1, 1, 255);

    CImgDisplay main_disp(mask, "PatchMatch Image Inpainting");

    int prev_x = -1, prev_y = -1;
    char prev_key = 0;

    int brush_radius = max(HALF_SIZE, static_cast<int>(0.03 * min_dimension));

    FillMode curr_mode = FillMode::FILL;

    // Handle user events on the image canvas
    while (!main_disp.is_closed()) {
        main_disp.wait();

        if (main_disp.is_key()) {
            char curr_key = main_disp.key();

            switch (main_disp.key()) {
                case TOGGLE_FILL_ON:
                    curr_mode = FillMode::FILL;
                    break;
                case TOGGLE_ERASE_ON:
                    curr_mode = FillMode::ERASE;
                    break;
                case TOGGLE_BRUSH_RAD_INCR:
                    brush_radius = min(min(image.width(), image.height()), static_cast<int>(ceil((1 + BRUSH_SIZE_MULT) * brush_radius)));
                    break;
                case TOGGLE_BRUSH_RAD_DECR:
                    brush_radius = max(HALF_SIZE, static_cast<int>(floor((1 - BRUSH_SIZE_MULT) * brush_radius)));
                    break;
                case TOGGLE_SAVE:
                    if (main_disp.is_keyCTRLLEFT())
                        main_disp.close();
                    break;
            }

            prev_key = curr_key;
        } else {
            prev_key = 0;
        }

        const int y = main_disp.mouse_y(), x = main_disp.mouse_x();

        if (main_disp.button() && inBounds(x, y, width, height)) {
            if (prev_x >= 0 && prev_y >= 0) {
                // Interpolate points between (prev_x, prev_y) and (x, y)
                int dx = x - prev_x;
                int dy = y - prev_y;
                int steps = max(abs(dx), abs(dy));

                for (int i = 0; i <= steps; i++) {
                    int inter_x = prev_x + i * dx / steps;
                    int inter_y = prev_y + i * dy / steps;
                    maskFillPatch(mask, masked_image, image, inter_x, inter_y, curr_mode, brush_radius);
                }
            } else {
                maskFillPatch(mask, masked_image, image, x, y, curr_mode, brush_radius);
            }

            prev_x = x;
            prev_y = y;
        } else {
            prev_x = prev_y = -1;
        }
        
        CImg<unsigned char> display_image = masked_image;
        if (inBounds(x, y, width, height))
            display_image.draw_ellipse(x, y, brush_radius, brush_radius, 0, COLOR_WHITE, 1, ~0U);

        // Draw the hotkeys information text
        display_image.draw_text(10, 10, "1: Fill On", COLOR_WHITE, COLOR_BLACK);
        display_image.draw_text(10, 30, "2: Erase On", COLOR_WHITE, COLOR_BLACK);
        display_image.draw_text(10, 50, "x: Increase Brush Radius", COLOR_WHITE, COLOR_BLACK);
        display_image.draw_text(10, 70, "z: Decrease Brush Radius", COLOR_WHITE, COLOR_BLACK);
        display_image.draw_text(10, 90, "ctrl+s: Save", COLOR_WHITE, COLOR_BLACK);
        
        main_disp.display(display_image);
    }

    return 0;
}