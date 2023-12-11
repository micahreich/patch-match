#include "patch_match.h"

#include <omp.h>

#include <cassert>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>

#include "cycle_timer.h"
#include "patch_match_utils.h"

using namespace std;
using namespace cv;

void printProgressBar(int step, int total_steps)
{
    const int bar_width = 60;
    float progress = static_cast<float>(step) / total_steps;

    cout << "[";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
}

Rect maskBoundingRect(mask_t &mask)
{
    int min_row = mask.rows - 1;
    int max_row = 0;
    int min_col = mask.cols - 1;
    int max_col = 0;

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask.at<bool>(r, c)) {
                min_row = min(min_row, r);
                max_row = max(max_row, r);
                min_col = min(min_col, c);
                max_col = max(max_col, c);
            }
        }
    }

    return Rect(min_col, min_row, max_col - min_col + 1, max_row - min_row + 1);
}

void PatchMatchInpainter::initPyramids(image_t image, mask_t mask)
{
    // Allocate space for all levels of the pyramid
    shift_map_pyramid = new shift_map_t[params.n_levels];
    distance_map_pyramid = new distance_map_t[params.n_levels];
    texture_pyramid = new texture_t[params.n_levels];
    mask_pyramid = new mask_t[params.n_levels];
    image_pyramid = new image_t[params.n_levels];
    dilated_mask_pyramid = new mask_t[params.n_levels];
    hole_region_pyramid = new Rect[params.n_levels];

    // Convert image to grayscale
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);

    // Compute gradients
    Mat gradient_x, gradient_y;

    Sobel(gray_image, gradient_x, CV_16S, 1, 0);
    Mat abs_gradient_x = abs(gradient_x);
    normalize(abs_gradient_x, abs_gradient_x, 0, 255, cv::NORM_MINMAX, CV_8U);

    Sobel(gray_image, gradient_y, CV_16S, 0, 1);
    Mat abs_gradient_y = abs(gradient_y);
    normalize(abs_gradient_y, abs_gradient_y, 0, 255, cv::NORM_MINMAX, CV_8U);

    Mat blurred_abs_gradient_x, blurred_abs_gradient_y;

    int texture_blur_sidelen = 1 + pow(2, params.n_levels - 1);

    blur(abs_gradient_x, blurred_abs_gradient_x, Size(texture_blur_sidelen, texture_blur_sidelen));
    blur(abs_gradient_y, blurred_abs_gradient_y, Size(texture_blur_sidelen, texture_blur_sidelen));

    // Stack abs_gradient_x and abs_gradient_y along the 3rd dimension to form
    // the texture matrix
    Mat texture;
    Mat channels[2] = {blurred_abs_gradient_x, blurred_abs_gradient_y};
    merge(channels, 2, texture);

    image_pyramid[0] = image;
    mask_pyramid[0] = mask;
    texture_pyramid[0] = texture;
    dilate(mask, dilated_mask_pyramid[0], patch_dilation_element);

    if (debug_mode_XXX) {
        destroyAllWindows();

        // Display the first values in image_pyramid, mask_pyramid, texture_pyramid, and dilated_mask_pyramid
        namedWindow("Image Pyramid - Level 0", WINDOW_NORMAL);
        imshow("Image Pyramid - Level 0", image_pyramid[0]);

        namedWindow("Mask Pyramid - Level 0", WINDOW_NORMAL);
        Mat normalized_mask_pyramid;
        mask_pyramid[0].convertTo(normalized_mask_pyramid, CV_8UC1, 255);
        imshow("Mask Pyramid - Level 0", normalized_mask_pyramid);

        // Split the texture pyramid into two separate channels
        vector<Mat> split_texture;
        split(texture_pyramid[0], split_texture);

        // Display the x gradients
        namedWindow("Texture Pyramid - X Gradients - Level 0", WINDOW_NORMAL);
        imshow("Texture Pyramid - X Gradients - Level 0", split_texture[0]);

        // Display the y gradients
        namedWindow("Texture Pyramid - Y Gradients - Level 0", WINDOW_NORMAL);
        imshow("Texture Pyramid - Y Gradients - Level 0", split_texture[1]);

        namedWindow("Dilated Mask Pyramid - Level 0", WINDOW_NORMAL);
        Mat normalized_dilated_mask_pyramid;
        dilated_mask_pyramid[0].convertTo(normalized_dilated_mask_pyramid, CV_8UC1, 255);
        imshow("Dilated Mask Pyramid - Level 0", normalized_dilated_mask_pyramid);

        waitKey(0);
    }

    for (unsigned int i = 1; i < params.n_levels; ++i) {
        image_t previous_image = image_pyramid[i - 1], previous_image_blurred, next_level_image;

        GaussianBlur(previous_image, previous_image_blurred, Size(3, 3), 1, 1);
        resize(previous_image_blurred, next_level_image, Size(), 0.5, 0.5, INTER_LINEAR);
        image_pyramid[i] = next_level_image;

        int multiplier = pow(2, i);

        texture_t next_level_texture;
        resize(texture_pyramid[0], next_level_texture, Size(), 1.f / multiplier, 1.f / multiplier, INTER_NEAREST);
        texture_pyramid[i] = next_level_texture;

        mask_t next_level_mask;
        resize(mask_pyramid[i - 1], next_level_mask, Size(), 0.5, 0.5, INTER_NEAREST);
        mask_pyramid[i] = next_level_mask;

        dilate(mask_pyramid[i], dilated_mask_pyramid[i], patch_dilation_element);

        if (debug_mode_XXX) {
            destroyAllWindows();

            // Display the values in image_pyramid, mask_pyramid, texture_pyramid, and dilated_mask_pyramid
            namedWindow("Image Pyramid - Level " + to_string(i), WINDOW_NORMAL);
            imshow("Image Pyramid - Level " + to_string(i), image_pyramid[i]);

            namedWindow("Mask Pyramid - Level " + to_string(i), WINDOW_NORMAL);
            Mat normalized_mask_pyramid;
            mask_pyramid[i].convertTo(normalized_mask_pyramid, CV_8UC1, 255);
            imshow("Mask Pyramid - Level " + to_string(i), normalized_mask_pyramid);

            // Split the texture pyramid into two separate channels
            vector<Mat> split_texture;
            split(texture_pyramid[i], split_texture);

            // Display the x gradients
            namedWindow("Texture Pyramid - X Gradients - Level " + to_string(i), WINDOW_NORMAL);
            imshow("Texture Pyramid - X Gradients - Level " + to_string(i), split_texture[0]);

            // Display the y gradients
            namedWindow("Texture Pyramid - Y Gradients - Level " + to_string(i), WINDOW_NORMAL);
            imshow("Texture Pyramid - Y Gradients - Level " + to_string(i), split_texture[1]);

            namedWindow("Dilated Mask Pyramid - Level " + to_string(i), WINDOW_NORMAL);
            Mat normalized_dilated_mask_pyramid;
            dilated_mask_pyramid[i].convertTo(normalized_dilated_mask_pyramid, CV_8UC1, 255);
            imshow("Dilated Mask Pyramid - Level " + to_string(i), normalized_dilated_mask_pyramid);

            waitKey(0);
        }
    }

    // Pad all images, textures, and masks with half_size pixels of padding
    for (unsigned int i = 0; i < params.n_levels; i++) {
        image_t padded_img, padded_img_expanded_dtype;
        copyMakeBorder(image_pyramid[i], padded_img, params.half_size, params.half_size, params.half_size,
                       params.half_size, BORDER_REPLICATE);
        // Convert to 32-bit signed integer -- this is needed for future computations which will do element-wise
        // arithmetic on image slices, and we don't want it to overflow/clip at intermediate results
        padded_img.convertTo(padded_img_expanded_dtype, CV_32S);
        image_pyramid[i] = padded_img_expanded_dtype;

        texture_t padded_texture, padded_texture_expanded_dtype;
        copyMakeBorder(texture_pyramid[i], padded_texture, params.half_size, params.half_size, params.half_size,
                       params.half_size, BORDER_REPLICATE);
        padded_texture.convertTo(padded_texture_expanded_dtype, CV_32S);  // Convert to 32-bit signed integer
        texture_pyramid[i] = padded_texture_expanded_dtype;

        mask_t padded_mask;
        copyMakeBorder(mask_pyramid[i], padded_mask, params.half_size, params.half_size, params.half_size,
                       params.half_size, BORDER_CONSTANT, 0);
        mask_pyramid[i] = padded_mask;

        mask_t padded_dilated_mask;
        copyMakeBorder(dilated_mask_pyramid[i], padded_dilated_mask, params.half_size, params.half_size,
                       params.half_size, params.half_size, BORDER_CONSTANT, 0);
        dilated_mask_pyramid[i] = padded_dilated_mask;

        Rect bounding_rect = maskBoundingRect(padded_dilated_mask);
        hole_region_pyramid[i] = bounding_rect;
    }

    // Initialize the coarsest level of the shift map pyramid
    int coarse_image_h = this->image_pyramid[params.n_levels - 1].rows;
    int coarse_image_w = this->image_pyramid[params.n_levels - 1].cols;
    mask_t corase_dilated_mask = this->dilated_mask_pyramid[params.n_levels - 1];

    shift_map_t coarse_shift_map = shift_map_t::zeros(coarse_image_h, coarse_image_w, CV_32SC2);

    for (int r = 0; r < coarse_image_h; r++) {
        for (int c = 0; c < coarse_image_w; c++) {
            Vec2i current_index = Vec2i(r, c);
            Vec2i candidate_index(current_index);

            while (corase_dilated_mask.at<bool>(candidate_index[0], candidate_index[1])) {
                int random_row = generateRandomInt(params.half_size, coarse_image_h - params.half_size);
                int random_col = generateRandomInt(params.half_size, coarse_image_w - params.half_size);
                candidate_index = Vec2i(random_row, random_col);
            }

            coarse_shift_map.at<Vec2i>(r, c) = candidate_index - current_index;
        }
    }

    this->shift_map_pyramid[params.n_levels - 1] = coarse_shift_map;

    // Initialize the coarsest level of the distance pyramid to zeros
    this->distance_map_pyramid[params.n_levels - 1] = distance_map_t::zeros(coarse_image_h, coarse_image_w, CV_32FC1);
}

float PatchMatchInpainter::patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB, AlgorithmStage stage,
                                         double &time, image_t& image, texture_t& texture,
                                         optional<reference_wrapper<mask_t>> init_shrinking_mask = nullopt,
                                         string marker = "")
{
    // TODO @dkrajews: Turns out patchDistance is responsible for a shit load of the runtime ...

    // If on initialization, we mask out the A and B regions using the shrinking_mask (as it appears in region A)
    mask_t* shrinking_mask = stage == AlgorithmStage::INITIALIZATION ? &init_shrinking_mask->get() : nullptr;
//    if (stage == AlgorithmStage::INITIALIZATION) {
//        assert(init_shrinking_mask != nullopt);
//        mask_t shrinking_mask = init_shrinking_mask->get();
//    }
//    else {
//        assert(init_shrinking_mask == nullopt);
//    }

    // Get the current level's image and texture pyramids
//    image_t image = this->image_pyramid[pyramid_idx];
//    texture_t texture = this->texture_pyramid[pyramid_idx];
//
    size_t image_h = image.rows, image_w = image.cols;
//
    Rect regionA = patchRegion(centerA, image_h, image_w, false);

//    Rect regionB = patchRegion(centerB, image_h, image_w, false);
    // TODO @mreich: look at region "intersection"

    float unoccluded_patch_area = params.patch_area;

    float ssd_image = 0.f;
    float ssd_texture = 0.f;

////     This loop over the image region is heavily optimized, hence all the pointers and direct memory accesses
// //    rather than making a copy of the image region and looping over that

    for (int i = 0; i < regionA.height; ++i) {
        // Pointers to the beginning of the i-th row in the respective regions
        auto *row_ptr_imageA = image.ptr<Vec3i>(centerA[0] - params.half_size + i) + centerA[1] - params.half_size;
        auto *row_ptr_imageB = image.ptr<Vec3i>(centerB[0] - params.half_size + i) + centerB[1] - params.half_size;

        auto *row_ptr_textureA = texture.ptr<Vec2i>(centerA[0] - params.half_size + i) + centerA[1] - params.half_size;
        auto *row_ptr_textureB = texture.ptr<Vec2i>(centerB[0] - params.half_size + i) + centerB[1] - params.half_size;

        const uchar *row_ptr_maskA = nullptr;
        if (stage == AlgorithmStage::INITIALIZATION) {
            row_ptr_maskA =
                (*shrinking_mask).ptr<uchar>(centerA[0] - params.half_size + i) + centerA[1] - params.half_size;
        }
        else {
            row_ptr_maskA = this->patch_size_zeros.ptr<uchar>(0);
        }

//        if(regionA.width != 5) {
//          printf("Width: %d\n", regionA.width);
//        }
        for (int j = 0; j < regionA.width; ++j) {
            if (row_ptr_maskA[j] > 0) {
                --unoccluded_patch_area;
                continue;
            }

            Vec3i &pixelA = row_ptr_imageA[j];
            Vec3i &pixelB = row_ptr_imageB[j];

            Vec2i &textureA = row_ptr_textureA[j];
            Vec2i &textureB = row_ptr_textureB[j];

//          ssd_image +=
//              (pixelA[0] - pixelB[0]) * (pixelA[0] - pixelB[0]) +
//              (pixelA[1] - pixelB[1]) * (pixelA[1] - pixelB[1]) +
//              (pixelA[2] - pixelB[2]) * (pixelA[2] - pixelB[2]);
//
//          ssd_texture +=
//              (textureA[0] - textureB[0]) * (textureA[0] - textureB[0]) +
//              (textureA[1] - textureB[1]) * (textureA[1] - textureB[1]);

            Vec3i pixel_diff = pixelA - pixelB;
            Vec2i texture_diff = textureA - textureB;

            ssd_image +=
                (pixel_diff[0] * pixel_diff[0]) + (pixel_diff[1] * pixel_diff[1]) + (pixel_diff[2] * pixel_diff[2]);

            ssd_texture += (texture_diff[0] * texture_diff[0]) + (texture_diff[1] * texture_diff[1]);
        }
    }

    float distance = (1.f / unoccluded_patch_area) * (ssd_image + params.lambda * ssd_texture);
    return distance;
}

void PatchMatchInpainter::reconstructImage(int pyramid_idx, AlgorithmStage stage,
                                           optional<reference_wrapper<mask_t>> init_boundary_mask = nullopt,
                                           optional<reference_wrapper<mask_t>> init_shrinking_mask = nullopt)
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
    Rect bounding_box = this->hole_region_pyramid[pyramid_idx];

    image_t updated_image = image.clone();
    texture_t updated_texture = texture.clone();

    mask_t mask = this->mask_pyramid[pyramid_idx];
    distance_map_t distance_map = this->distance_map_pyramid[pyramid_idx];
    shift_map_t shift_map = this->shift_map_pyramid[pyramid_idx];

    size_t image_h = image.rows, image_w = image.cols;

    for (int r = bounding_box.y; r < bounding_box.y + bounding_box.height; r++) {
        for (int c = bounding_box.x; c < bounding_box.x + bounding_box.width; c++) {
            if (stage == AlgorithmStage::INITIALIZATION && !boundary_mask.at<bool>(r, c))
                continue;
            else if (stage != AlgorithmStage::INITIALIZATION && !mask.at<bool>(r, c))
                continue;

            Rect region = patchRegion(Vec2i(r, c), image_h, image_w, true);
            unsigned int patch_area = region.area();

            Vec2i best_neighborhood_pixel = Vec2i(r, c);
            float best_neighborhood_distance = distance_map.at<float>(r, c);

            // Find the 75th percentile distance (of those unmasked distances, if in initialization)
            vector<double> region_distances(patch_area, 0.f);
            vector<Vec2i> pixels(patch_area, Vec2i(0, 0));

            unsigned int k = 0;
            for (int i = region.y; i < region.y + region.height; i++) {
                for (int j = region.x; j < region.x + region.width; j++) {
                    float dist = distance_map.at<float>(i, j);
                    Vec2i px = Vec2i(i, j);

                    switch (stage) {
                        case AlgorithmStage::INITIALIZATION:
                            if (!shrinking_mask.at<bool>(i, j)) {
                                pixels[k] = Vec2i(i, j);
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

            // On final stage, we fill in the pixel at (r, c) with the color/texture from the best neighborhood pixel's
            // shifted area (pixel in neighborhood with lowest distance value)
            if (stage == AlgorithmStage::FINAL) {
                Vec2i shift = shift_map.at<Vec2i>(best_neighborhood_pixel[0], best_neighborhood_pixel[1]);

                updated_image.at<Vec3i>(r, c) = image.at<Vec3i>(r + shift[0], c + shift[1]);
                updated_texture.at<Vec2i>(r, c) = texture.at<Vec2i>(r + shift[0], c + shift[1]);

                continue;
            }
            // On non-final stage, we weight the pixels in the neighborhood by
            // their distance values and take a weighted average of the shifted
            // pixels to fill in color/texture
            vector<double> scores(region_distances);

            unsigned int n_excluded = patch_area - k;
            unsigned int q = static_cast<unsigned int>(n_excluded + 0.75f * k);

            assert(q < region_distances.size() && q >= 0);

            std::nth_element(region_distances.begin(), region_distances.begin() + q, region_distances.end());
            float sigma_p = max(1e-6, region_distances[q]);

            // Find each pixel's weight and take a weighted sum of pixels in the neighborhood
            float scores_sum = 0.f;
            for (int l = 0; l < k; l++) {
                scores[l] = exp(-scores[l] / (2 * sigma_p * sigma_p));
                scores_sum += scores[l];
            }

            Vec3d image_pixel = Vec3f(0, 0, 0);
            Vec2d texture_pixel = Vec2f(0, 0);

            for (int l = 0; l < k; l++) {
                float pixel_weight = scores[l] / scores_sum;
                Vec2i shift = shift_map.at<Vec2i>(pixels[l][0], pixels[l][1]);

                image_pixel += scores[l] * image.at<Vec3i>(r + shift[0], c + shift[1]);
                texture_pixel += scores[l] * texture.at<Vec2i>(r + shift[0], c + shift[1]);
            }

            image_pixel /= scores_sum;
            texture_pixel /= scores_sum;

            Vec3b final_image_pixel = Vec3b(saturate_cast<uchar>(image_pixel[0]), saturate_cast<uchar>(image_pixel[1]),
                                            saturate_cast<uchar>(image_pixel[2]));

            Vec2b final_texture_pixel =
                Vec2b(saturate_cast<uchar>(texture_pixel[0]), saturate_cast<uchar>(texture_pixel[1]));

            updated_image.at<Vec3i>(r, c) = final_image_pixel;
            updated_texture.at<Vec2i>(r, c) = final_texture_pixel;
        }
    }

    this->image_pyramid[pyramid_idx] = updated_image;
    this->texture_pyramid[pyramid_idx] = updated_texture;
}

vector<int> jumpFloodRadii(int pyramid_idx, int max_dimension)
{
    vector<int> radii = {max_dimension};
    while (radii.back() > 1) {
        radii.push_back(radii.back() / 2);
    }

    // Perform JFA + 2 algorithm by adding extra radii of 2 and 1 in at the end
    radii.push_back(2);
    radii.push_back(1);

    return radii;
}

void PatchMatchInpainter::approximateNearestNeighbor(int pyramid_idx, AlgorithmStage stage, double &patch_distance_time,
                                                     optional<reference_wrapper<mask_t>> init_boundary_mask = nullopt,
                                                     optional<reference_wrapper<mask_t>> init_shrinking_mask = nullopt)
{
    // TODO @dkrajews: when profiling ANN, also profile patchDistance. Inside patchDistance, I want to see if when I do
    // something like image_regionA = image(regionA) -- which copies the image content -- significantly slows stuff down
    // or not. If it does, then we should try to avoid copying the image content and instead just loop through the
    // matrix directly computing values rather than using opencv's element-wise stuff

    // We should do an extensive profiling of the function and see which parts take up the most time

    mask_t boundary_mask, shrinking_mask;
    if (stage == AlgorithmStage::INITIALIZATION) {
        assert(init_boundary_mask != nullopt);
        assert(init_shrinking_mask != nullopt);

        boundary_mask = init_boundary_mask->get();
        shrinking_mask = init_shrinking_mask->get();
    }
    else {
        assert(init_boundary_mask == nullopt);
        assert(init_shrinking_mask == nullopt);
    }

    // Get the current level's image and texture pyramids
    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];
    Rect bounding_box = this->hole_region_pyramid[pyramid_idx];

    mask_t mask = this->mask_pyramid[pyramid_idx];
    mask_t dilated_mask = this->dilated_mask_pyramid[pyramid_idx];

    distance_map_t distance_map = this->distance_map_pyramid[pyramid_idx];

    shift_map_t shift_map = this->shift_map_pyramid[pyramid_idx].clone();
    shift_map_t updated_shift_map = shift_map.clone();

    // TODO @dkrajews: investigate if things are being copied here correctly
    shift_map_t *active_shift_map = &updated_shift_map;
    shift_map_t *prev_shift_map = &shift_map;

    size_t image_h = image.rows, image_w = image.cols;
    size_t max_image_dim = max(image_h, image_w);

    distance_map_t updated_distance_map = distance_map.clone();

    vector<int> jump_flood_radii = jumpFloodRadii(pyramid_idx, max_image_dim);

    double total_patch_distance_time = 0.f;
    double pdt;

    for (int k = 0; k < params.n_iters_jfa * jump_flood_radii.size(); k++) {
        int idx = k % jump_flood_radii.size();
        int jump_flood_radius = jump_flood_radii[idx];

        int radii_offsets[3] = {-jump_flood_radius, 0, jump_flood_radius};

//        #pragma omp parallel for collapse(2) // @dkrajews: this is the main parallelization point in ANN
        for (int r = bounding_box.y; r < bounding_box.y + bounding_box.height; r++) {
            for (int c = bounding_box.x; c < bounding_box.x + bounding_box.width; c++) {
                if (stage == AlgorithmStage::INITIALIZATION && !boundary_mask.at<bool>(r, c))
                    continue;
                else if (stage == AlgorithmStage::NORMAL && !dilated_mask.at<bool>(r, c))
                    continue;

                Vec2i curr_coordinate = Vec2i(r, c);
                Vec2i best_shift = prev_shift_map->at<Vec2i>(r, c);

                float best_distance = patchDistance(pyramid_idx, curr_coordinate, curr_coordinate + best_shift, stage,
                                                    pdt, image, texture, init_shrinking_mask, "marker");
                total_patch_distance_time += pdt;

                // Iterate through all 9 neighbors at the current jump flood radius
                for (auto dr : radii_offsets) {
                    for (auto dc : radii_offsets) {
                        Vec2i partner_coordinate = Vec2i(r + dr, c + dc);
                        if (!inBounds(partner_coordinate[0], partner_coordinate[1], image_h, image_w, params.half_size))
                            continue;

                        Vec2i candidate_shift = prev_shift_map->at<Vec2i>(partner_coordinate[0], partner_coordinate[1]);
                        Vec2i candidate_coordinate = curr_coordinate + candidate_shift;
                        if (!inBounds(candidate_coordinate[0], candidate_coordinate[1], image_h, image_w,
                                      params.half_size))
                            continue;

                        float candidate_distance = patchDistance(pyramid_idx, curr_coordinate, candidate_coordinate,
                                                                 stage, pdt, image, texture, init_shrinking_mask);
                        total_patch_distance_time += pdt;

                        if (!dilated_mask.at<bool>(candidate_coordinate[0], candidate_coordinate[1]) &&
                            candidate_distance < best_distance) {
                            best_distance = candidate_distance;
                            best_shift = candidate_shift;
                        }
                    }
                }

                // Random search step, exponential backoff from original offset
                float alpha = 1.f;
                Vec2i original_shift = best_shift;

                while (alpha * max_image_dim >= 1) {
                    int random_row = generateRandomInt(-alpha * max_image_dim, alpha * max_image_dim);
                    int random_col = generateRandomInt(-alpha * max_image_dim, alpha * max_image_dim);

                    Vec2i random_shift_offset = Vec2i(random_row, random_col);
                    Vec2i candidate_coordinate = curr_coordinate + original_shift + random_shift_offset;

                    if (!inBounds(candidate_coordinate[0], candidate_coordinate[1], image_h, image_w, params.half_size))
                        continue;

                    float candidate_distance = patchDistance(pyramid_idx, curr_coordinate, candidate_coordinate, stage,
                                                             pdt, image, texture, init_shrinking_mask);
                    total_patch_distance_time += pdt;

                    if (!dilated_mask.at<bool>(candidate_coordinate[0], candidate_coordinate[1]) &&
                        candidate_distance < best_distance) {
                        best_distance = candidate_distance;
                        best_shift = original_shift + random_shift_offset;
                    }

                    alpha *= 0.5f;
                }

                // Update the active shift map and distance map
                active_shift_map->at<Vec2i>(r, c) = best_shift;
                updated_distance_map.at<float>(r, c) = best_distance;
            }
        }

        // Swap active and previous shift map pointers between jump flood iterations
        std::swap(active_shift_map, prev_shift_map);
    }

    patch_distance_time = total_patch_distance_time;

    // Place the most recently updated shift map back into the pyramid
    this->shift_map_pyramid[pyramid_idx] = *prev_shift_map;
    this->distance_map_pyramid[pyramid_idx] = updated_distance_map;
}

void PatchMatchInpainter::annHelper(int r, int c, image_t& image, texture_t& texture, int pyramid_idx, vector<int>& jump_flood_radii, mask_t& dilated_mask,
               shift_map_t *active_shift_map, shift_map_t *prev_shift_map, shift_map_t& updated_shift_map, distance_map_t& updated_distance_map,
               double &patch_distance_time, int& idx, int& jump_flood_radius, int radii_offsets[],
               optional<reference_wrapper<mask_t>> init_boundary_mask = nullopt,
               optional<reference_wrapper<mask_t>> init_shrinking_mask = nullopt) {

  size_t image_h = image.rows, image_w = image.cols;
  size_t max_image_dim = max(image_h, image_w);
  double total_patch_distance_time = 0.f;
  double pdt;



//        #pragma omp parallel for collapse(2) // @dkrajews: this is the main parallelization point in ANN
    if (!dilated_mask.at<bool>(r, c))
      return;

    Vec2i curr_coordinate = Vec2i(r, c);
    Vec2i best_shift = prev_shift_map->at<Vec2i>(r, c);

    float best_distance = patchDistance(pyramid_idx, curr_coordinate, curr_coordinate + best_shift, AlgorithmStage::NORMAL,
                                        pdt, image, texture,  init_shrinking_mask, "marker");
    total_patch_distance_time += pdt;

    // Iterate through all 9 neighbors at the current jump flood radius
    for (int dr_i = 0; dr_i < 3; dr_i++) {
      auto dr = radii_offsets[dr_i];
      for (int dc_i = 0; dc_i < 3; dc_i++) {
        auto dc = radii_offsets[dc_i];
        Vec2i partner_coordinate = Vec2i(r + dr, c + dc);
        if (!inBounds(partner_coordinate[0], partner_coordinate[1], image_h, image_w, params.half_size))
          continue;

        Vec2i candidate_shift = prev_shift_map->at<Vec2i>(partner_coordinate[0], partner_coordinate[1]);
        Vec2i candidate_coordinate = curr_coordinate + candidate_shift;
        if (!inBounds(candidate_coordinate[0], candidate_coordinate[1], image_h, image_w,
                      params.half_size))
          continue;

        float candidate_distance = patchDistance(pyramid_idx, curr_coordinate, candidate_coordinate,
                                                 AlgorithmStage::NORMAL, pdt, image, texture,  init_shrinking_mask);
        total_patch_distance_time += pdt;

        if (!dilated_mask.at<bool>(candidate_coordinate[0], candidate_coordinate[1]) &&
            candidate_distance < best_distance) {
          best_distance = candidate_distance;
          best_shift = candidate_shift;
        }
      }
    }

    // Random search step, exponential backoff from original offset
    float alpha = 1.f;
    Vec2i original_shift = best_shift;

    while (alpha * max_image_dim >= 1) {
      int random_row = generateRandomInt(-alpha * max_image_dim, alpha * max_image_dim);
      int random_col = generateRandomInt(-alpha * max_image_dim, alpha * max_image_dim);

      Vec2i random_shift_offset = Vec2i(random_row, random_col);
      Vec2i candidate_coordinate = curr_coordinate + original_shift + random_shift_offset;

      if (!inBounds(candidate_coordinate[0], candidate_coordinate[1], image_h, image_w, params.half_size))
        continue;

      float candidate_distance = patchDistance(pyramid_idx, curr_coordinate, candidate_coordinate, AlgorithmStage::NORMAL,
                                               pdt, image, texture,  init_shrinking_mask);
      total_patch_distance_time += pdt;

      if (!dilated_mask.at<bool>(candidate_coordinate[0], candidate_coordinate[1]) &&
          candidate_distance < best_distance) {
        best_distance = candidate_distance;
        best_shift = original_shift + random_shift_offset;
      }

      alpha *= 0.5f;
    }

    // Update the active shift map and distance map
    active_shift_map->at<Vec2i>(r, c) = best_shift;
    updated_distance_map.at<float>(r, c) = best_distance;

    // Swap active and previous shift map pointers between jump flood iterations
//    std::swap(active_shift_map, prev_shift_map);

}

void PatchMatchInpainter::reconstructionHelper(int r, int c, image_t& image, texture_t& texture, int pyramid_idx, mask_t mask, vector<int>& jump_flood_radii, mask_t& dilated_mask,
                          shift_map_t *active_shift_map, shift_map_t *prev_shift_map, shift_map_t& updated_shift_map, distance_map_t& updated_distance_map,
                          double &patch_distance_time, distance_map_t& distance_map, shift_map_t& shift_map,
                          image_t& updated_image, texture_t& updated_texture)
{
  size_t image_h = image.rows, image_w = image.cols;
  size_t max_image_dim = max(image_h, image_w);

  mask_t boundary_mask, shrinking_mask;

  if (!mask.at<bool>(r, c))
    return;

  Rect region = patchRegion(Vec2i(r, c), image_h, image_w, true);
  unsigned int patch_area = region.area();

  Vec2i best_neighborhood_pixel = Vec2i(r, c);
  float best_neighborhood_distance = distance_map.at<float>(r, c);

  // Find the 75th percentile distance (of those unmasked distances, if in initialization)
  vector<double> region_distances(patch_area, 0.f);
  vector<Vec2i> pixels(patch_area, Vec2i(0, 0));

  unsigned int k = 0;
  for (int i = region.y; i < region.y + region.height; i++) {
    for (int j = region.x; j < region.x + region.width; j++) {
      float dist = distance_map.at<float>(i, j);
      Vec2i px = Vec2i(i, j);
      pixels[k] = px;
      region_distances[k] = dist;
      k++;
//      switch (stage) {
//        case AlgorithmStage::INITIALIZATION:
//          if (!shrinking_mask.at<bool>(i, j)) {
//            pixels[k] = Vec2i(i, j);
//            region_distances[k] = dist;
//            k++;
//          }
//
//          break;
//        case AlgorithmStage::NORMAL:
//
//
//          break;
//        case AlgorithmStage::FINAL:
//          if (dist < best_neighborhood_distance) {
//            best_neighborhood_distance = dist;
//            best_neighborhood_pixel = px;
//          }
//
//          break;
//      }
    }
  }

  // On final stage, we fill in the pixel at (r, c) with the color/texture from the best neighborhood pixel's
  // shifted area (pixel in neighborhood with lowest distance value)
//  if (stage == AlgorithmStage::FINAL) {
//    Vec2i shift = shift_map.at<Vec2i>(best_neighborhood_pixel[0], best_neighborhood_pixel[1]);
//
//    updated_image.at<Vec3i>(r, c) = image.at<Vec3i>(r + shift[0], c + shift[1]);
//    updated_texture.at<Vec2i>(r, c) = texture.at<Vec2i>(r + shift[0], c + shift[1]);
//
//    continue;
//  }
  // On non-final stage, we weight the pixels in the neighborhood by
  // their distance values and take a weighted average of the shifted
  // pixels to fill in color/texture
  vector<double> scores(region_distances);

  unsigned int n_excluded = patch_area - k;
  unsigned int q = static_cast<unsigned int>(n_excluded + 0.75f * k);

  assert(q < region_distances.size() && q >= 0);

  std::nth_element(region_distances.begin(), region_distances.begin() + q, region_distances.end());
  float sigma_p = max(1e-6, region_distances[q]);

  // Find each pixel's weight and take a weighted sum of pixels in the neighborhood
  float scores_sum = 0.f;
  for (int l = 0; l < k; l++) {
    scores[l] = exp(-scores[l] / (2 * sigma_p * sigma_p));
    scores_sum += scores[l];
  }

  Vec3d image_pixel = Vec3f(0, 0, 0);
  Vec2d texture_pixel = Vec2f(0, 0);

  for (int l = 0; l < k; l++) {
    float pixel_weight = scores[l] / scores_sum;
    Vec2i shift = shift_map.at<Vec2i>(pixels[l][0], pixels[l][1]);

    image_pixel += scores[l] * image.at<Vec3i>(r + shift[0], c + shift[1]);
    texture_pixel += scores[l] * texture.at<Vec2i>(r + shift[0], c + shift[1]);
  }

  image_pixel /= scores_sum;
  texture_pixel /= scores_sum;

  Vec3b final_image_pixel = Vec3b(saturate_cast<uchar>(image_pixel[0]), saturate_cast<uchar>(image_pixel[1]),
                                  saturate_cast<uchar>(image_pixel[2]));

  Vec2b final_texture_pixel =
      Vec2b(saturate_cast<uchar>(texture_pixel[0]), saturate_cast<uchar>(texture_pixel[1]));

  updated_image.at<Vec3i>(r, c) = final_image_pixel;
  updated_texture.at<Vec2i>(r, c) = final_texture_pixel;
}

bool nonEmptyMask(mask_t &mask)
{
    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask.at<bool>(r, c)) return true;
        }
    }

    return false;
}

void boundaryMask(mask_t &mask, mask_t &dst, optional<reference_wrapper<mask_t>> eroded_mask = nullopt)
{
    mask_t eroded;

    if (eroded_mask == nullopt) {
        Mat structure_elem = getStructuringElement(MORPH_CROSS, Size(3, 3));
        erode(mask, eroded, structure_elem);
    }
    else {
        eroded = eroded_mask->get();
    }

    subtract(mask, eroded, dst);
}

void PatchMatchInpainter::onionPeelInit()
{
    // Initialize the shrinking mask to be the initialization mask
    int pyramid_idx = params.n_levels - 1;
    mask_t shrinking_mask = this->dilated_mask_pyramid[pyramid_idx];

    int onion_peel_level = 0;
    while (++onion_peel_level, nonEmptyMask(shrinking_mask)) {
        // Get the boundary of the shrinking mask
        mask_t eroded_shrinking_mask;
        Mat structure_elem = getStructuringElement(MORPH_CROSS, Size(3, 3));
        erode(shrinking_mask, eroded_shrinking_mask, structure_elem);

        mask_t boundary_shrinking_mask;
        boundaryMask(shrinking_mask, boundary_shrinking_mask,
                     optional<reference_wrapper<mask_t>>(ref(eroded_shrinking_mask)));

        // Overlay the combined_mask on the image from level n_levels - 1 in the image pyramid
        if (debug_mode_XXX) {
            printf("Onion Peel Level: %d\n", onion_peel_level);
            destroyAllWindows();

            // Create a 3-channel image to display both masks
            Mat combined_mask = Mat::zeros(shrinking_mask.size(), CV_8UC3);

            // Color the boundary mask pure red
            Mat boundary_channels[] = {Mat::zeros(shrinking_mask.size(), CV_8U), boundary_shrinking_mask * 255,
                                       Mat::zeros(shrinking_mask.size(), CV_8U)};
            Mat boundary_mask;
            merge(boundary_channels, 3, boundary_mask);

            // Color the eroded mask pure green
            Mat eroded_channels[] = {Mat::zeros(shrinking_mask.size(), CV_8U), Mat::zeros(shrinking_mask.size(), CV_8U),
                                     eroded_shrinking_mask * 255};
            Mat eroded_mask;
            merge(eroded_channels, 3, eroded_mask);

            // Combine the masks
            add(boundary_mask, eroded_mask, combined_mask);

            // Get the image from level n_levels - 1 in the image pyramid
            image_t image = this->image_pyramid[params.n_levels - 1];

            // Convert the image to 8U before doing any arithmetic
            image_t image_8U;
            image.convertTo(image_8U, CV_8U);

            // Create a new matrix for the overlaid image
            Mat overlaid_image;

            // Overlay the combined mask on the image
            addWeighted(image_8U, 0.5, combined_mask, 0.5, 0.0, overlaid_image);

            // Display the overlayed image
            imshow("Overlayed Image", overlaid_image);
            waitKey(0);
        }

        // Perform ANN search for pixels on the shrinking mask boundary
        double patch_distance_time;
        approximateNearestNeighbor(pyramid_idx, AlgorithmStage::INITIALIZATION, patch_distance_time,
                                   optional<reference_wrapper<mask_t>>(ref(boundary_shrinking_mask)),
                                   optional<reference_wrapper<mask_t>>(ref(shrinking_mask)));

        if (onion_peel_level > params.half_size) {
            // If progressed enough into the initialization to start filling in the actual mask values, reconstruct
            // boundary vals
            reconstructImage(pyramid_idx, AlgorithmStage::INITIALIZATION,
                             optional<reference_wrapper<mask_t>>(ref(boundary_shrinking_mask)),
                             optional<reference_wrapper<mask_t>>(ref(shrinking_mask)));
        }

        // Update the shrinking mask to be the eroded version of itself
        shrinking_mask = eroded_shrinking_mask;
    }
}

image_t PatchMatchInpainter::inpaint()
{
//  #pragma omp parallel {
//  #prahma
//  }
    if (debug_mode_XXX) {
        printf("Beginning onion peel initialization .....\n");
    }
    auto onion_peel_init_start = CycleTimer::currentSeconds();

    int total_steps = params.n_levels + 1;

    printProgressBar(0, total_steps);

    onionPeelInit();

    auto onion_peel_init_end = CycleTimer::currentSeconds();
    auto onion_peel_init_duration = onion_peel_init_end - onion_peel_init_start;
    this->timing_stats.initialization_time = onion_peel_init_duration;

    for (int l = params.n_levels - 1; l >= 0; --l) {
        printProgressBar(params.n_levels - l, total_steps);

        auto level_start = CycleTimer::currentSeconds();

        this->timing_stats.ann_times.push_back(vector<double>());
        this->timing_stats.reconstruction_times.push_back(vector<double>());

        Rect bounding_box = this->hole_region_pyramid[l];

        mask_t boundary_mask, shrinking_mask;

        // Get the current level's image and texture pyramids
        image_t image = this->image_pyramid[l];
        texture_t texture = this->texture_pyramid[l];

        mask_t mask = this->mask_pyramid[l];
        mask_t dilated_mask = this->dilated_mask_pyramid[l];

        distance_map_t distance_map = this->distance_map_pyramid[l];

        shift_map_t shift_map = this->shift_map_pyramid[l].clone();
        shift_map_t updated_shift_map = shift_map.clone();

        // TODO @dkrajews: investigate if things are being copied here correctly
        shift_map_t *active_shift_map = &updated_shift_map;
        shift_map_t *prev_shift_map = &shift_map;

        size_t image_h = image.rows, image_w = image.cols;
        size_t max_image_dim = max(image_h, image_w);

        distance_map_t updated_distance_map = distance_map.clone();

        vector<int> jump_flood_radii = jumpFloodRadii(l, max_image_dim);

        double total_patch_distance_time = 0.f;
        double pdt, patch_distance_time;

        image_t updated_image = image.clone();
        texture_t updated_texture = texture.clone();
//        vector<int> counter(10);

//      approximateNearestNeighbor(l, AlgorithmStage::NORMAL, patch_distance_time);
//      reconstructImage(l, AlgorithmStage::NORMAL);

      if(l != 0) {
        approximateNearestNeighbor(l, AlgorithmStage::NORMAL, patch_distance_time);
        reconstructImage(l, AlgorithmStage::NORMAL);
      } else {

//       #pragma omp parallel default(shared)
//         {

//             for (int k = 0; k < params.n_iters; ++k) {
//     //           #pragma omp for
//               // init ANN stuff

//               // TODO: Put jumpflood iterations here
//               for (int j = 0; j < params.n_iters_jfa * jump_flood_radii.size(); j++) {
//                 int idx = j % jump_flood_radii.size();
//                 int jump_flood_radius = jump_flood_radii[idx];

//                 int radii_offsets[3] = {-jump_flood_radius, 0, jump_flood_radius};


//                 #pragma omp for collapse(2) schedule(dynamic)
//                 for (int r = bounding_box.y; r < bounding_box.y + bounding_box.height; r++) {
//                   for (int c = bounding_box.x; c < bounding_box.x + bounding_box.width; c++) {
//                     // run ANN and Reconstruction for individual pixel
//                     annHelper(r, c, image, texture, l, jump_flood_radii, dilated_mask, active_shift_map,
//                               prev_shift_map, updated_shift_map, updated_distance_map,
//                               patch_distance_time, idx, jump_flood_radius, radii_offsets);
// //                    counter[omp_get_thread_num()]++;

//                   }
//                 }

//                 #pragma omp single
//                 {
// //                  for(int x = 0; x < 10; x++) {
// //                    printf("%d: %d\n", x, counter[x]);
// //                  }
// //                  printf("Area: %d\n", bounding_box.area());
// //                  printf("=========================\n");
// //                  counter = vector<int>(10);
//                   std::swap(active_shift_map, prev_shift_map);
//                 };
//               }

//               #pragma omp barrier

//               #pragma omp single
//               {
//                 this->shift_map_pyramid[l] = *prev_shift_map;
//                 this->distance_map_pyramid[l] = updated_distance_map;
//               };


// //              #pragma omp for collapse(2)
// //              for (int r = bounding_box.y; r < bounding_box.y + bounding_box.height; r++) {
// //                for (int c = bounding_box.x; c < bounding_box.x + bounding_box.width; c++) {
// //                  // run ANN and Reconstruction for individual pixel
// //                  reconstructionHelper(r, c, image, texture, l, mask, jump_flood_radii, dilated_mask,
// //                                       active_shift_map, prev_shift_map, updated_shift_map, updated_distance_map,
// //                                       patch_distance_time, distance_map, shift_map, updated_image, updated_texture);
// //
// //                }
// //              }

//               #pragma omp single
//               {
//                 reconstructImage(l, AlgorithmStage::NORMAL);
// //                this->image_pyramid[l] = updated_image;
// //                this->texture_pyramid[l] = updated_texture;
//               };




//             }
//         }
      }




      //  #pragma omp barrier ??




//            if (debug_mode) printf("\tk = %d\n", k);
//
//            auto ann_start = CycleTimer::currentSeconds();
//
//            // Perform ANN search
//            double patch_distance_time;
//            approximateNearestNeighbor(l, AlgorithmStage::NORMAL, patch_distance_time);
//            // printf("patch_distance_time: %f ms\n", 1000.f * patch_distance_time);
//
//            auto ann_end = CycleTimer::currentSeconds();
//            auto ann_duration = ann_end - ann_start;
//            this->timing_stats.ann_times[(params.n_levels - 1) - l].push_back(ann_duration);
//
//            auto reconstruction_start = CycleTimer::currentSeconds();
//
//            // Perform image and texture reconstruction based on updated shift map
//            reconstructImage(l, AlgorithmStage::NORMAL);
//
//            auto reconstruction_end = CycleTimer::currentSeconds();
//            auto reconstruction_duration = reconstruction_end - reconstruction_start;
//            this->timing_stats.reconstruction_times[(params.n_levels - 1) - l].push_back(reconstruction_duration);
//        }

        // TODO @dkrajews: PARALLELIZE
        /**
         * for every pixel:
         *  for i in range(10):
         *      for j in range(n jump flood levels)
         *          ann search at jump flood radius j
         *          sync threads
         *
         *      reconstruct pixel
         */


        // TODO @dkrajews: add timing code and something in TimingStats for timing upsampling of distance map and shift
        // map and also time the upsampling reconstruction
        if (l == 0) {
            reconstructImage(l, AlgorithmStage::FINAL);
        }
        else {
            shift_map_t upsampled_shift_map = upsampleZeroPad(this->shift_map_pyramid[l], params.half_size, true);

            distance_map_t upsampled_distance_map =
                upsampleZeroPad(this->distance_map_pyramid[l], params.half_size, false);

            this->shift_map_pyramid[l - 1] = upsampled_shift_map;
            this->distance_map_pyramid[l - 1] = upsampled_distance_map;

            reconstructImage(l - 1, AlgorithmStage::NORMAL);
        }

        if (write_levels) {
            char filename[64];
            snprintf(filename, sizeof(filename), "inpaint-lvl-%d.png", l);
            imwrite(filename, this->image_pyramid[l]);
        }

        auto level_end = CycleTimer::currentSeconds();
        auto level_duration = level_end - level_start;
        this->timing_stats.level_times.push_back(level_duration);
    }

    printProgressBar(total_steps, total_steps);
    cout << endl << endl;

    // Crop final image back to initial size
    image_t final_image = this->image_pyramid[0].clone();
    Rect crop(params.half_size, params.half_size, final_image.cols - 2 * params.half_size,
              final_image.rows - 2 * params.half_size);
    final_image = final_image(crop);

    if (write_levels) {
        char filename[64];
        imwrite("inpaint-final.png", final_image);
    }

    this->timing_stats.prettyPrint();

    return final_image;
}

PatchMatchInpainter::PatchMatchInpainter(image_t image, mask_t mask, PatchMatchParams params = PatchMatchParams())
    : params(params)
{
    //    srand(time(0));

    this->timing_stats = TimingStats();
    this->patch_dilation_element =
        getStructuringElement(MORPH_RECT, Size(this->params.patch_size, this->params.patch_size));
    this->patch_size_ones = Mat::ones(this->params.patch_size, this->params.patch_size, CV_8UC1);
    this->patch_size_zeros = Mat::zeros(this->params.patch_size, this->params.patch_size, CV_8UC1);

    // Initialize all image, texture, etc. pyramids given the initial image and mask

    // TODO @dkrajews: is this gonna copy image and mask? should it? should we
    // use references?
    auto start = CycleTimer::currentSeconds();

    initPyramids(image, mask);

    auto end = CycleTimer::currentSeconds();

    this->timing_stats.pyramid_build_time = end - start;
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