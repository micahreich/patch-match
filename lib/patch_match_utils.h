#ifndef  __UTILS_H__
#define  __UTILS_H__

#include <stdio.h>
#include <string.h>
#include <cmath>
#include <vector>
#include <random>
#include <opencv2/core.hpp>
#include <ctime>
#include <cstdlib>

typedef cv::Mat shift_map_t;
typedef cv::Mat distance_map_t;
typedef cv::Mat texture_t;
typedef cv::Mat mask_t;
typedef cv::Mat image_t;

// Function to check if a pixel is in bounds
static bool inBounds(int y, int x, int height, int width, int half_size=0) {
    return (x >= half_size && x < width - half_size && y >= half_size && y < height - half_size);
}

// Function to generate a random integer in [lb, ub) range
static int generateRandomInt(int lb, int ub) {
    return lb + rand() % (ub - lb);
}

// Function to generate a random float in [lb, ub) range
static float generateRandomFloat(float lb, float ub) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = ub - lb;
    float r = random * diff;
    return lb + r;
}



#endif