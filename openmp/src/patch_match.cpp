#include <cassert>

#include "utils.h"
// #include <iostream>
#include "patch_match.h"
#include <opencv2/opencv.hpp>
#include <optional>

using namespace std;
using namespace cv;

void PatchMatchInpainter::initPyramids(image_t image, mask_t mask)
{
    // Allocate space for all levels of the pyramid
    shift_map_pyramid = new shift_map_t[n_levels];
    distance_map_pyramid = new distance_map_t[n_levels];
    texture_pyramid = new texture_t[n_levels];
    mask_pyramid = new mask_t[n_levels];
    image_pyramid = new image_t[n_levels];
    dilated_mask_pyramid = new mask_t[n_levels];

    // Convert image to grayscale
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    // Compute gradients
    Mat gradient_x, gradient_y, abs_gradient_x, abs_gradient_y;

    Sobel(gray_image, gradient_x, CV_16S, 1, 0);
    convertScaleAbs(gradient_x, abs_gradient_x);

    Sobel(gray_image, gradient_y, CV_16S, 0, 1);
    convertScaleAbs(gradient_y, abs_gradient_y);
    
    Mat blurred_abs_gradient_x, blurred_abs_gradient_y;

    int texture_blur_sidelen = 1 + pow(2, n_levels - 1);

    blur(abs_gradient_x, blurred_abs_gradient_x, Size(texture_blur_sidelen, texture_blur_sidelen));
    blur(abs_gradient_y, blurred_abs_gradient_y, Size(texture_blur_sidelen, texture_blur_sidelen));
    
    // Stack abs_gradient_x and abs_gradient_y along the 3rd dimension to form the texture matrix
    Mat texture;
    vector<Mat> channels = {blurred_abs_gradient_x, blurred_abs_gradient_y};
    merge(channels, texture);


    image_pyramid[0] = image;
    mask_pyramid[0] = mask;
    texture_pyramid[0] = texture;
    dilate(mask, dilated_mask_pyramid[0], patch_dilation_element);

    
    for(unsigned int i = 1; i < n_levels; ++i) {
        image_t previous_image = image_pyramid[i-1], previous_image_blurred, next_level_image;

        GaussianBlur(previous_image, previous_image_blurred, Size(3, 3), 1, 1);
        resize(previous_image_blurred, next_level_image, Size(), 0.5, 0.5, INTER_LINEAR);
        image_pyramid[i] = next_level_image;

        int multiplier = pow(2, i);

        texture_t next_level_texture;
        resize(texture_pyramid[0], next_level_texture, Size(), 1.f/multiplier, 1.f/multiplier, INTER_NEAREST);
        texture_pyramid[i] = next_level_texture;

        mask_t next_level_mask;
        resize(mask_pyramid[i-1], next_level_mask, Size(), 0.5, 0.5, INTER_NEAREST);
        mask_pyramid[i] = next_level_mask;

        dilate(mask_pyramid[i], dilated_mask_pyramid[i], patch_dilation_element);
    }

    // Pad all images, textures, and masks with half_size pixels of padding
    for(unsigned int i = 1; i < n_levels; i++) {
        image_t padded_img;
        copyMakeBorder(image_pyramid[i], padded_img, half_size, half_size, half_size, half_size, BORDER_REPLICATE);
        image_pyramid[i] = padded_img;

        texture_t padded_texture;
        copyMakeBorder(texture_pyramid[i], padded_texture, half_size, half_size, half_size, half_size, BORDER_REPLICATE);
        texture_pyramid[i] = padded_texture;

        mask_t padded_mask;
        copyMakeBorder(mask_pyramid[i], padded_mask, half_size, half_size, half_size, half_size, BORDER_CONSTANT, 0);
        mask_pyramid[i] = padded_mask;

        mask_t padded_dilated_mask;
        copyMakeBorder(dilated_mask_pyramid[i], padded_dilated_mask, half_size, half_size, half_size, half_size, BORDER_CONSTANT, 0);
        dilated_mask_pyramid[i] = padded_dilated_mask;
    }

    // Initialize the coarsest level of the shift map pyramid
    int coarse_image_h = this->image_pyramid[n_levels - 1].rows;
    int coarse_image_w = this->image_pyramid[n_levels - 1].cols;

    shift_map_t coarse_shift_map = shift_map_t::zeros(coarse_image_h, coarse_image_w, CV_32SC2);

    for (int r = 0; r < coarse_image_h; r++) {
        for (int c = 0; c < coarse_image_w; c++) {
            Vec2i current_index = Vec2i(r, c);
            Vec2i candidate_index(current_index);

            while (this->dilated_mask_pyramid[n_levels - 1].at<bool>(candidate_index[0], candidate_index[1]))
            {
                int random_row = generateRandomInt(half_size, coarse_image_h - half_size);
                int random_col = generateRandomInt(half_size, coarse_image_w - half_size);
                candidate_index = Vec2i(random_row, random_col);
            }

            coarse_shift_map.at<Vec2i>(r, c) = candidate_index - current_index;
        }
    }

    this->shift_map_pyramid[n_levels - 1] = coarse_shift_map;
    
    // Initialize the coarsest level of the distance pyramid to zeros
    this->distance_map_pyramid[n_levels - 1] = distance_map_t::zeros(coarse_image_h, coarse_image_w, CV_32FC1);
}


float PatchMatchInpainter::patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB, AlgorithmStage stage,
                                         optional<reference_wrapper<mask_t>> init_shrinking_mask=nullopt)
{
    // If on initialization, we mask out the A and B regions using the shrinking_mask (as it appears in region A)
    mask_t shrinking_mask;
    if (stage == AlgorithmStage::INITIALIZATION) {
        assert(init_shrinking_mask != nullopt);
        shrinking_mask = init_shrinking_mask->get();
    }

    // Get the current level's image and texture pyramids
    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];

    size_t image_h = image.rows, image_w = image.cols;
    assert(inBounds(centerA[0], centerA[1], image_w, image_h, half_size)); // Should always be in bounds (outside padding)

    Rect regionA = patchRegion(centerA, image_h, image_w, false);
    Rect regionB = patchRegion(centerB, image_h, image_w, false);

    float unoccluded_patch_area = patch_size * patch_size;
    
    Mat image_regionA = image(regionA);
    Mat image_regionB = image(regionB);
    Mat texture_regionA = texture(regionA);
    Mat texture_regionB = texture(regionB);

    Mat image_region_difference = image_regionA - image_regionB; // Sum of squared differences
    image_region_difference.mul(image_region_difference);

    Mat texture_region_difference = texture_regionA - texture_regionB;
    texture_region_difference.mul(texture_region_difference);

    // If masked, calculate how many pixels are unmasked in the region and mask the regions
    if (stage == AlgorithmStage::INITIALIZATION) {        
        Scalar n_occluded = sum(shrinking_mask(regionA));
        unoccluded_patch_area = regionA.area() - n_occluded[0];

        assert(unoccluded_patch_area > 0);

        Mat mask_region = shrinking_mask(regionA);

        image_region_difference.setTo(Scalar::all(0), mask_region);
        texture_region_difference.setTo(Scalar::all(0), mask_region);
    }

    int ssd_image = sum(image_region_difference)[0];
    int ssd_texture = sum(texture_region_difference)[0];

    return 1.f / unoccluded_patch_area * (ssd_image + lambda * ssd_texture);
}

image_t PatchMatchInpainter::reconstructImage(int pyramid_idx, AlgorithmStage stage,
                                              optional<reference_wrapper<mask_t>> init_boundary_mask=nullopt,
                                              optional<reference_wrapper<mask_t>> init_shrinking_mask=nullopt)
{
    mask_t boundary_mask, shrinking_mask;
    if (stage == AlgorithmStage::INITIALIZATION) {
        assert(init_boundary_mask != nullopt);
        assert(init_shrinking_mask != nullopt);

        boundary_mask = init_boundary_mask->get();
        shrinking_mask = init_shrinking_mask->get();
    }

    // Get the current level's image and texture pyramids
    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];
    mask_t mask = this->mask_pyramid[pyramid_idx];
    distance_map_t distance_map = this->distance_map_pyramid[pyramid_idx];
    shift_map_t shift_map = this->shift_map_pyramid[pyramid_idx];

    size_t image_h = image.rows, image_w = image.cols;
    unsigned int patch_area = patch_size * patch_size;
    
    for (int r = half_size; r < image_h - half_size; r++) {
        for (int c = half_size; c < image_w - half_size; c++) {
            if (stage == AlgorithmStage::INITIALIZATION && !boundary_mask.at<bool>(r, c)) continue;
            else if (stage == AlgorithmStage::NORMAL && !mask.at<bool>(r, c)) continue;

            Vec2i best_neighborhood_pixel = Vec2i(r, c);
            float best_neighborhood_distance = distance_map.at<float>(r, c);

            // Find the 75th percentile distance (of those unmasked distances, if in initialization)
            vector<float> region_distances(patch_area, 0.f);
            vector<Vec2i> pixels(patch_area);

            Rect region = patchRegion(Vec2i(r, c), image_h, image_w, true);

            unsigned int k = 0;
            for (int i = region.y; i < region.height; i++) {
                for (int j = region.x; j < region.width; j++) {
                    float dist = distance_map.at<float>(i, j);
                    Vec2i px = Vec2i(i, j);

                    switch (stage) {
                        case AlgorithmStage::INITIALIZATION:
                            if (!shrinking_mask.at<bool>(i, j)) {
                                pixels[k] = px;
                                region_distances[k] = dist;
                                k++;
                            }

                            break;
                        case AlgorithmStage::NORMAL:
                            pixels[k] = px;
                            region_distances[k] = dist;
                            k++;

                            break;
                        case AlgorithmStage::FINAL:
                            if (dist < best_neighborhood_distance) {
                                best_neighborhood_distance = dist;
                                best_neighborhood_pixel = px;
                            }

                            break;
                    }

                }
            }

            // On final stage, we fill in the pixel at (r, c) with the color/texture from the best 
            // neighborhood pixel's shifted area (pixel in neighborhood with lowest distance value)
            if (stage == AlgorithmStage::FINAL) {
                Vec2i shift = shift_map.at<Vec2i>(best_neighborhood_pixel[0], best_neighborhood_pixel[1]);

                image.at<Vec3b>(r, c) = image.at<Vec3b>(r + shift[0], c + shift[1]);
                texture.at<Vec2f>(r, c) = texture.at<Vec2i>(r + shift[0], c + shift[1]);

                continue;
            }

            // On non-final stage, we weight the pixels in the neighborhood by their distance values
            // and take a weighted average of the shifted pixels to fill in color/texture
            vector<float> scores(region_distances);

            unsigned int n_excluded = patch_area - k;
            unsigned int q = static_cast<unsigned int>(n_excluded + 0.75f * k);
            
            nth_element(region_distances.begin(), region_distances.begin() + q, region_distances.end());
            float sigma_p = region_distances[q];

            // Find each pixel's weight and take a weighted sum of pixels in the neighborhood
            float scores_sum = 0.f;
            for (int i = 0; i < k; i++) {
                scores[i] = expf(-scores[i] / (2 * sigma_p * sigma_p));
                scores_sum += scores[i];
            }

            Vec3b image_pixel;
            Vec2f texture_pixel;

            for (int i = 0; i < k; i++) {
                float pixel_weight = scores[i] / scores_sum;

                Vec2i shift = shift_map.at<Vec2i>(pixels[k][0], pixels[k][1]);
                
                image_pixel += pixel_weight * image.at<Vec3b>(r + shift[0], c + shift[1]);
                texture_pixel += pixel_weight * texture.at<Vec2i>(r + shift[0], c + shift[1]);
            }

            image.at<Vec3b>(r, c) = image_pixel;
            texture.at<Vec2f>(r, c) = texture_pixel;
        }
    }
}


bool nonEmptyMask(mask_t& mask)
{
    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask.at<bool>(r, c)) return true;
        }
    }

    return false;
}


mask_t boundaryMask(mask_t& mask, mask_t& dst, optional<reference_wrapper<mask_t>> eroded_mask=nullopt)
{
    mask_t eroded;

    if (eroded_mask == nullopt) {
        Mat structure_elem = getStructuringElement(MORPH_CROSS, Size(3, 3));
        erode(mask, eroded, structure_elem);
    } else {
        eroded = eroded_mask->get();
    }
    
    subtract(mask, eroded, dst);
}


void PatchMatchInpainter::onionPeelInit()
{
    // Initialize the shrinking mask to be the initialization mask
    mask_t shrinking_mask = this->dilated_mask_pyramid[0];

    while (nonEmptyMask(shrinking_mask)) {
        // Get the boundary of the shrinking mask
        mask_t eroded_shrinking_mask;
        Mat structure_elem = getStructuringElement(MORPH_CROSS, Size(3, 3));
        erode(shrinking_mask, eroded_shrinking_mask, structure_elem);

        mask_t boundary_shrinking_mask;
        boundaryMask(shrinking_mask, boundary_shrinking_mask, eroded_shrinking_mask);

        // Perform ANN search for pixels on the shrinking mask boundary
        // approximateNearestNeighbor()
    }
}

PatchMatchInpainter::PatchMatchInpainter(unsigned int n_levels, unsigned int patch_size,
                                         image_t image, mask_t mask) 
{
    this->n_levels = n_levels;
    this->patch_size = patch_size;
    this->half_size = patch_size / 2;
    this->patch_dilation_element = getStructuringElement(MORPH_RECT, Size(patch_size, patch_size));

    // Initialize all image, texture, etc. pyramids given the initial image and mask
    initPyramids(image, mask);
}

PatchMatchInpainter::~PatchMatchInpainter()
{
    delete[] shift_map_pyramid;
    delete[] distance_map_pyramid;
    delete[] texture_pyramid;
    delete[] mask_pyramid;
    delete[] image_pyramid;
    delete[] dilated_mask_pyramid;
}