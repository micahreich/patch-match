#include <stdint.h>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define PATCH_SIZE 7
#define HALF_SIZE PATCH_SIZE / 2
#define BRUSH_SIZE_MULT 0.2f

#define TOGGLE_FILL_ON '1'         // ASCII for 1
#define TOGGLE_ERASE_ON '2'        // ASCII for 2
#define TOGGLE_BRUSH_RAD_INCR 'x'  // ASCII for x
#define TOGGLE_BRUSH_RAD_DECR 'z'  // ASCII for z
#define TOGGLE_SAVE 's'

const unsigned char MASK_COLOR[] = {0, 255, 0};
const unsigned char COLOR_WHITE[] = {255, 255, 255};
const unsigned char COLOR_BLACK[] = {0, 0, 0};

#include <unistd.h>

#include "patch_match.h"
#include "utils.h"

using namespace std;

const string WINDOW_NAME = "PhotoShop - CS418";

enum FillMode { ERASE = 0, FILL = 1 };

cv::Point lastPoint(-1, -1);
cv::Point currentMousePos(-1, -1);
int brushRadius = 15;  // Initial brush radius
FillMode curr_mode = FillMode::FILL;
cv::Mat drawingLayer;
int n_levels = 0, patch_size = 0, lambda = 0;
int minimum_levels = 1, minimum_patch_size = 5, minimum_lambda = 1;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    currentMousePos = {x, y};
    if (event == cv::EVENT_LBUTTONDOWN) {
        lastPoint = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
        if (lastPoint.x != -1) {
            cv::Scalar color = curr_mode == FillMode::ERASE
                                   ? cv::Scalar(0, 0, 0, 0)
                                   : cv::Scalar(57, 255, 20, 1);  // Transparent for eraser, red for drawing
            // int thickness = curr_mode == FillMode::ERASE ? brushRadius * 2 : brushRadius; // Larger thickness for
            // eraser for easier erasing
            cv::line(drawingLayer, lastPoint, cv::Point(x, y), color, brushRadius, cv::LINE_AA, 0);
            lastPoint = cv::Point(x, y);
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        lastPoint = cv::Point(-1, -1);
    }
}

void onLevelsChange(int new_value, void* userdata) { n_levels = minimum_levels + new_value; }

void onPatchSizeChange(int new_value, void* userdata) { patch_size = minimum_patch_size + new_value; }

void onLambdaChange(int new_value, void* userdata) { lambda = minimum_lambda + new_value; }

int main(int argc, char* argv[]) {
    // Read in user image with -i command line flag, or use lena.png as default
    const char* image_path = "src/lena.png";
    int opt;
    while ((opt = getopt(argc, argv, "di:")) != -1) {
        switch (opt) {
            case 'd':
                debug_mode = true;
                break;
            case 'i':
                image_path = optarg;
                break;
            default:
                fprintf(stderr, "Usage: %s [-i image_path] [-d]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    cv::Mat image = cv::imread("./src/lena.png");
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    drawingLayer = cv::Mat::zeros(image.size(), CV_8UC4);
    cv::namedWindow(WINDOW_NAME);
    cv::setMouseCallback(WINDOW_NAME, onMouse, &image);
    cv::moveWindow(WINDOW_NAME, 100, 100);

    cv::createTrackbar("Num Levels", WINDOW_NAME, nullptr, 10, onLevelsChange);
    cv::createTrackbar("Patch Size", WINDOW_NAME, nullptr, 25, onPatchSizeChange);
    cv::createTrackbar("Lambda", WINDOW_NAME, nullptr, 10, onLambdaChange);

    cv::setTrackbarPos("Num Levels", WINDOW_NAME, minimum_levels);
    cv::setTrackbarPos("Patch Size", WINDOW_NAME, minimum_patch_size);
    cv::setTrackbarPos("Lambda", WINDOW_NAME, minimum_lambda);

    while (true) {
        // cv::imshow("Paint on Image", image);
        cv::Mat displayImage = image.clone();
        cv::Mat tempImage = image.clone();
        cv::cvtColor(image, tempImage, cv::COLOR_BGR2BGRA);  // Convert image to have an alpha channel
        cv::addWeighted(tempImage, 1.0, drawingLayer, 1.0, 0, displayImage);
        if (currentMousePos.x != -1 && currentMousePos.y != -1) {
            cv::Scalar outlineColor = curr_mode == FillMode::ERASE
                                          ? cv::Scalar(0, 0, 255, 255)
                                          : cv::Scalar(255, 0, 0, 255);  // Blue for eraser, red for drawing
            cv::circle(displayImage, currentMousePos, brushRadius / 2, outlineColor, 1);
        }
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Point textOrg(10, 30);  // Position of the text
        cv::Point offset(0, 20);
        cv::Scalar textColor(0, 255, 0);  // Green color

        cv::putText(displayImage, "Keys: [x]/[z] to change brush size", textOrg, fontFace, fontScale, textColor,
                    thickness);
        cv::putText(displayImage, "Keys: [1]/[2] to change fill mdoe", textOrg + offset, fontFace, fontScale, textColor,
                    thickness);

        cv::imshow(WINDOW_NAME, displayImage);

        // cv::setTrackbarPos("Patch Size", "Paint on Image", patch_size);

        char key = cv::waitKey(1);
        if (key == 27)  // ESC key to exit
            break;
        else if (key == 'x')  // Increase brush radius
            brushRadius += 2;
        else if (key == 'z' && brushRadius > 1)  // Decrease brush radius
            brushRadius -= 2;
        else if (key == TOGGLE_FILL_ON) {
            curr_mode = FillMode::FILL;
        } else if (key == TOGGLE_ERASE_ON) {
            curr_mode = FillMode::ERASE;
        }
    }

    // cv::imwrite("modified_image.jpg", image);
    cv::destroyAllWindows();

    // Get the drawn mask and convert it to a cv::Mat of 0s and 1s
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::cvtColor(drawingLayer, mask, cv::COLOR_BGR2GRAY);
    cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
    cv::Mat mask_mat = mask > 0;

    // Prints the mask to visually inspect and ensure its correct
    // for(int i = 0; i < image.rows; i++) {
    //     for(int j = 0; j < image.cols; j++) {
    //         printf("%d ", mask_mat.at<bool>(i, j));
    //     }
    //     printf("\n");
    // }

    PatchMatchParams params = PatchMatchParams();
    params.n_levels = n_levels;
    params.lambda = lambda;
    params.patch_size = patch_size;

    PatchMatchInpainter inpainter(params, image, mask_mat);

    // verify that reconstructed image works
    // CImg<unsigned char> reconstructed_image(height, width, 1, 3, 0);
    // cimg_forXY(reconstructed_image, x, y) {
    //     RGBPixel pixel = img_array(x, y);
    //     reconstructed_image(x, y, 0) = pixel.r;
    //     reconstructed_image(x, y, 1) = pixel.g;
    //     reconstructed_image(x, y, 2) = pixel.b;
    // }
    // CImgDisplay main_disp_2(reconstructed_image,"Reconstructed Image");
    // while (!main_disp_2.is_closed()) {
    //     main_disp_2.wait();
    // }

    return EXIT_SUCCESS;
}