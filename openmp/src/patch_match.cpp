#include <cassert>
// #include <iostream>
#include "patch_match.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <optional>

void PatchMatchInpainter::initPyramids(image_t image, mask_t mask)
{
    // Allocate space for all levels of the pyramid
    dimensions_pyramid = new Dimension[n_levels];
    shift_map_pyramid = new shift_map_t[n_levels];
    distance_map_pyramid = new distance_map_t[n_levels];
    texture_pyramid = new texture_t[n_levels];
    mask_pyramid = new mask_t[n_levels];
    image_pyramid = new image_t[n_levels];
    dilated_mask_pyramid = new mask_t[n_levels];

    // TODO: Write image to texture function
    texture_t image_texture = texture_t::zeros(image.rows, image.cols, CV_8UC1);

    // Convert image to grayscale
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);

    // Compute gradients
    cv::Mat gradient_x, gradient_y;
    cv::Sobel(gray_image, gradient_x, CV_32F, 1, 0);
    cv::Sobel(gray_image, gradient_y, CV_32F, 0, 1);

    // Compute absolute values of gradients
    cv::Mat abs_gradient_x, abs_gradient_y;
    cv::convertScaleAbs(gradient_x, abs_gradient_x);
    cv::convertScaleAbs(gradient_y, abs_gradient_y);

    // Initialize texture map
    texture_t texture = texture_t::zeros(image.rows, image.cols, CV_32FC2);

    // Calculate the finest level texture map from absolute values of image gradients
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            int sidelen = 1 + pow(2, n_levels - 1);
            int halflen = sidelen / 2;

            // Define texture region
            cv::Rect texture_region = cv::Rect(std::max(0, j - halflen), std::max(0, i - halflen), sidelen, sidelen) & cv::Rect(0, 0, image.cols, image.rows);

            // Compute texture
            cv::Scalar sum_gradient_x = cv::sum(abs_gradient_x(texture_region));
            cv::Scalar sum_gradient_y = cv::sum(abs_gradient_y(texture_region));

            float cardinality = sidelen * sidelen;

            texture.at<cv::Vec2f>(i, j)[0] = std::abs(sum_gradient_x[0] / cardinality);
            texture.at<cv::Vec2f>(i, j)[1] = std::abs(sum_gradient_y[0] / cardinality);
        }
    }

    image_texture = texture;

    // Show the mask as a binary image
    // cv::Mat binary_mask;
    // mask.convertTo(binary_mask, CV_8U);
    // cv::normalize(binary_mask, binary_mask, 0, 255, cv::NORM_MINMAX);
    // cv::imshow("Binary Mask", binary_mask);
    // cv::waitKey(0);
    
    // Display the first dimension of image texture
    // std::vector<cv::Mat> channels(2);
    // cv::split(texture, channels);

    // for (int i = 0; i < 2; ++i) {
    //     // Normalize the i-th channel to range [0, 255]
    //     cv::Mat normalized;
    //     cv::normalize(channels[i], normalized, 0, 255, cv::NORM_MINMAX);

    //     // Convert the floating-point matrix to an 8-bit grayscale image
    //     cv::Mat grayscale;
    //     normalized.convertTo(grayscale, CV_8U);

    //     // Display the image
    //     cv::imshow("Channel " + std::to_string(i), grayscale);
    // }

    // cv::waitKey(0);
    // cv::destroyAllWindows();

    image_pyramid[0] = image;
    mask_pyramid[0] = mask;
    texture_pyramid[0] = image_texture;
    cv::dilate(mask, dilated_mask_pyramid[0], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patch_size, patch_size)));
    
    for(unsigned int i = 1; i < n_levels; ++i) {
        image_t previous_image = image_pyramid[i-1];
        image_t next_level_image;
        cv::GaussianBlur(previous_image, next_level_image, cv::Size(0, 0), 1, 1);
        image_t next_level_image_downsampled;
        cv::resize(next_level_image, next_level_image_downsampled, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST_EXACT);
        image_pyramid[i] = next_level_image_downsampled;

        int multiplier = pow(2, i);

        texture_t next_level_texture;
        cv::resize(texture_pyramid[0], next_level_texture, cv::Size(), 1.f/multiplier, 1.f/multiplier, cv::INTER_NEAREST_EXACT);
        texture_pyramid[i] = next_level_texture;


        mask_t next_level_mask;
        cv::resize(mask_pyramid[i-1], next_level_mask, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST_EXACT);
        mask_pyramid[i] = next_level_mask;
        // cv::Mat boundary;
        //  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patch_size, patch_size))
        cv::dilate(mask_pyramid[i], dilated_mask_pyramid[i], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(patch_size, patch_size)));
        // cv::subtract(dilated_mask_pyramid[i], mask_pyramid[i], boundary);

        // cv::Mat binary_mask;
        // boundary.convertTo(binary_mask, CV_8U);
        // cv::normalize(binary_mask, binary_mask, 0, 255, cv::NORM_MINMAX);
        // cv::imshow("Binary Mask", binary_mask);
        // cv::waitKey(0);
    }

    for(unsigned int i = 1; i < n_levels; i++) {
        image_t padded_img;
        cv::copyMakeBorder(image_pyramid[i], padded_img, half_size, half_size, half_size, half_size, cv::BORDER_REPLICATE);
        image_pyramid[i] = padded_img;

        texture_t padded_texture;
        cv::copyMakeBorder(texture_pyramid[i], padded_texture, half_size, half_size, half_size, half_size, cv::BORDER_REPLICATE);
        texture_pyramid[i] = padded_texture;

        mask_t padded_mask;
        cv::copyMakeBorder(mask_pyramid[i], padded_mask, half_size, half_size, half_size, half_size, cv::BORDER_CONSTANT, 0);
        mask_pyramid[i] = padded_mask;
    }




    // for(int i = 0; i < n_levels; i++) {
    //     int image_h = image_pyramid[i].rows;
    //     int image_w = image_pyramid[i].cols;

    //     int mask_h = mask_pyramid[i].rows;
    //     int mask_w = mask_pyramid[i].cols;

    //     assert(image_h == mask_h);
    //     assert(image_w == mask_w);
    // }
    
    
}

float PatchMatchInpainter::patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB,
                                         std::optional<reference_wrapper<mask_t>> init_shrinking_mask)
{   
    mask_t shrinking_mask;
    if (pyramid_idx == 0) {
        assert(init_shrinking_mask != std::nullopt);
        shrinking_mask = init_shrinking_mask->get();
    }

    // Get the current level's image and texture pyramids
    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];

    size_t image_h = image.rows, image_w = image.cols;
    assert(inBounds(centerA.j, centerA.i, image_w, image_h, half_size)); // Should always be in bounds (outside padding)

    float occluded_patch_area = patch_size * patch_size;
    
    // If masked, calculate how many pixels are unmasked in the region
    if (pyramid_idx == 0) {
        ImageSliceCoords regionA = patchRegion(centerA, image_h, image_w);
        occluded_patch_area = 0.f;

        for (size_t r = regionA.row_start; r < regionA.row_end; r++) {
            for (size_t c = regionA.col_start; r < regionA.col_end; c++) {
                occluded_patch_area += !shrinking_mask.at<bool>(r, c);
            }
        }

        assert(occluded_patch_area > 0);
    }

    // Calculate the sum of squared differences between patches A and B in the RGB and texture image
    int ssd_image, ssd_texture = 0;

    for (int dr = -half_size; dr <= half_size; dr++) {
        for (int dc = -half_size; dc <= half_size; dc++) {
            int regionA_r = centerA.i + dr, regionA_c = centerA.j + dc;
            int regionB_r = centerB.i + dr, regionB_c = centerB.j + dc;

            if (pyramid_idx == 0 && shrinking_mask.at<bool>(regionA_r, regionA_c)) continue;

            cv::Vec3b rgb_difference = image.at<cv::Vec3b>(regionA_r, regionA_c) - image.at<cv::Vec3b>(regionB_r, regionB_c);
            rgb_difference = rgb_difference.mul(rgb_difference);

            cv::Vec2f texture_difference = texture.at<cv::Vec2f>(regionA_r, regionA_c) - texture.at<cv::Vec2f>(regionB_r, regionB_c);
            texture_difference = cv::Vec2f(texture_difference[0]*texture_difference[0], texture_difference[1]*texture_difference[1]);

            ssd_image += rgb_difference[0] + rgb_difference[1] + rgb_difference[1];
            ssd_texture += texture_difference[0] + texture_difference[1];
        }
    }

    return 1.f / occluded_patch_area * (ssd_image + lambda * ssd_texture);
}

image_t PatchMatchInpainter::reconstructImage(int pyramid_idx,
                                              std::optional<reference_wrapper<mask_t>> init_boundary_mask,
                                              std::optional<reference_wrapper<mask_t>> init_shrinking_mask)
{
    mask_t boundary_mask, shrinking_mask;
    if (pyramid_idx == 0) {
        assert(init_boundary_mask != std::nullopt);
        assert(init_shrinking_mask != std::nullopt);

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
            if (pyramid_idx == 0 && !boundary_mask.at<bool>(r, c)) continue;
            else if (pyramid_idx > 0 && !mask.at<bool>(r, c)) continue;

            // Find the 75th percentile distance (of those unmasked distances, if in initialization)
            vector<float> region_distances(patch_area, -1.f);
            vector<Vec2i> pixels(patch_area);

            ImageSliceCoords safe_region = patchRegion(Vec2i(r, c), image_h, image_w, true);

            unsigned int k = 0;
            for (int i = safe_region.row_start; i < safe_region.row_end; i++) {
                for (int j = safe_region.col_start; j < safe_region.col_end; j++) {
                    if (pyramid_idx > 0 || (pyramid_idx == 0 && !shrinking_mask.at<bool>(i, j))) {
                        pixels[k] = Vec2i(i, j);
                        region_distances[k] = distance_map.at<float>(i, j);
                        k++;
                    }
                }
            }

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

            cv::Vec3b image_pixel;
            cv::Vec2f texture_pixel;

            for (int i = 0; i < k; i++) {
                float entry_weight = scores[i] / scores_sum;

                cv::Vec2f shift = shift_map.at<cv::Vec2f>(pixels[k].i, pixels[k].j);
                
                image_pixel += entry_weight * image.at<cv::Vec3b>(r + shift[0], c + shift[1]);
                texture_pixel += entry_weight * texture.at<cv::Vec2f>(r + shift[0], c + shift[1]);
            }

            image.at<cv::Vec3b>(r, c) = image_pixel;
            texture.at<cv::Vec2f>(r, c) = texture_pixel;
        }
    }
}

void PatchMatchInpainter::onionPeelInit()
{

}

PatchMatchInpainter::PatchMatchInpainter(unsigned int n_levels, unsigned int patch_size,
                                         image_t image, mask_t mask) 
{
    this->n_levels = n_levels;
    this->patch_size = patch_size;
    this->half_size = patch_size/2;
    // Initialize all image, texture, etc. pyramids given the initial image and mask
    initPyramids(image, mask);

    // Initialize the level 0 shift map using random offsets for occluded pixels
    int last_level_index = n_levels - 1;
    int coarse_image_h = this->image_pyramid[last_level_index].rows;
    int coarse_image_w = this->image_pyramid[last_level_index].cols;
    for (int r = 0; r < coarse_image_h; r++) {
        for (int c = 0; c < coarse_image_w; c++) {
            Vec2i current_index = Vec2i(r, c);
            Vec2i candidate_index(current_index);

            while (this->dilated_mask_pyramid[last_level_index].at<bool>(candidate_index.i, candidate_index.j))
            {
                int random_row = rand() % (coarse_image_h - 2 * half_size) + half_size;
                int random_col = rand() % (coarse_image_w - 2 * half_size) + half_size;
                candidate_index = Vec2i(random_row, random_col);
            }
            this->shift_map_pyramid[last_level_index].at<cv::Vec2f>(r, c) = cv::Vec2f(candidate_index.i, candidate_index.j) - cv::Vec2f(current_index.i, current_index.j);
        }
    }
}
