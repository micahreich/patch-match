#include <cassert>

#include "patch_match.h"
#include "utils.h"

void PatchMatchInpainter::initPyramids(image_t image, mask_t mask)
{
    // Allocate space for all levels of the pyramid
    dimensions_pyramid = new Dimension[n_levels];
    shift_map_pyramid = new shift_map_t[n_levels];
    distance_map_pyramid = new distance_map_t[n_levels];
    texture_pyramid = new texture_t[n_levels];
    mask_pyramid = new mask_t[n_levels];
    image_pyramid = new image_t[n_levels];

    texture_t image_texture = texture_t(image.height, image.width);

    image_pyramid[0] = image;
    mask_pyramid[0] = mask;
    texture_pyramid[0] = image_texture;
    
    for(unsigned int i = 1; i < n_levels; ++i) {
        image_t previous_image = image_pyramid[i-1];
        image_t next_level_image = gaussianFilter(previous_image);
        image_t next_level_image_downsampled = image_t::downsample(next_level_image, 2);
        image_pyramid[i] = next_level_image_downsampled;

        int multiplier = pow(2, i);
        texture_t next_level_texture = texture_t::downsample(texture_pyramid[0], multiplier);
        texture_pyramid[i] = next_level_texture;

        mask_t next_level_mask = mask_t::downsample(mask_pyramid[i-1], 2);
        mask_pyramid[i] = next_level_mask;
    }

    for(unsigned int i = 1; i < n_levels; i++) {
        image_pyramid[i] = image_t::pad(image_pyramid[i], half_size, half_size);
        texture_pyramid[i] = texture_t::pad(texture_pyramid[i], half_size, half_size);
        mask_pyramid[i] = mask_t::pad(mask_pyramid[i], half_size, half_size, true);
    }

    for(int i = 0; i < n_levels; i++) {
        int image_h = image_pyramid[i].height;
        int image_w = image_pyramid[i].width;

        int mask_h = mask_pyramid[i].height;
        int mask_w = mask_pyramid[i].width;

        assert(image_h == mask_h);
        assert(image_w == mask_w);
    }
    

    // image_t coarse_image = image_pyramid[n_levels-1];
    // mask_t coarse_mask = mask_pyramid[n_levels-1];
    // mask_t dilated_coarse_mask = dilateMask(coarse_mask);
    // mask_t eroded_mask = erodeMask(dilated_coarse_mask);

    // shift_map_t shift_map(coarse_image.height, coarse_image.width);

    // unsigned int level_height = coarse_image.height, level_width = coarse_image.width;
    // for(unsigned int r = half_size; r < level_height-half_size; r++) {
    //     for(unsigned int c = half_size; c < level_width-half_size; c++) {
    //         int random_i = r;
    //         int random_j = c;

    //         while(dilated_coarse_mask(random_i, random_j)) {
    //             random_i = random_int(half_size, level_height-half_size);
    //             random_j = random_int(half_size, level_width - half_size);
    //         }

    //         int shift_i = random_i - r;
    //         int shift_j = random_j - c;
    //         shift_map(r, c) = Vec2i(shift_i, shift_j);
    //     }
    // }
    
}

float PatchMatchInpainter::patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB,
                                         optional<reference_wrapper<mask_t>> init_shrinking_mask=nullopt)
{   
    mask_t shrinking_mask;
    if (pyramid_idx == 0) {
        assert(init_shrinking_mask != nullopt);
        shrinking_mask = init_shrinking_mask->get();
    }

    // Get the current level's image and texture pyramids
    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];

    size_t image_h = image.height, image_w = image.width;
    assert(inBounds(centerA.j, centerA.i, image_w, image_h, half_size)); // Should always be in bounds (outside padding)

    float occluded_patch_area = patch_size * patch_size;
    
    // If masked, calculate how many pixels are unmasked in the region
    if (pyramid_idx == 0) {
        ImageSliceCoords regionA = patchRegion(centerA);
        occluded_patch_area = 0.f;

        for (size_t r = regionA.row_start; r < regionA.row_end; r++) {
            for (size_t c = regionA.col_start; r < regionA.col_end; c++) {
                occluded_patch_area += !shrinking_mask(r, c);
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

            if (pyramid_idx == 0 && shrinking_mask(regionA_r, regionA_c)) continue;

            RGBPixel rgb_difference = image(regionA_r, regionA_c) - image(regionB_r, regionB_c);
            rgb_difference *= rgb_difference;

            GradientPair texture_difference = texture(regionA_r, regionA_c) - texture(regionB_r, regionB_c);
            texture_difference *= texture_difference;

            ssd_image += rgb_difference.r + rgb_difference.g + rgb_difference.b;
            ssd_texture += texture_difference.grad_x + texture_difference.grad_y;
        }
    }

    return 1.f / occluded_patch_area * (ssd_image + lambda * ssd_texture);
}

image_t PatchMatchInpainter::reconstructImage(int pyramid_idx,
                                              optional<reference_wrapper<mask_t>> init_boundary_mask=nullopt,
                                              optional<reference_wrapper<mask_t>> init_shrinking_mask=nullopt)
{
    mask_t boundary_mask, shrinking_mask;
    if (pyramid_idx == 0) {
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

    unsigned int image_h = image.height, image_w = image.width;
    unsigned int patch_area = patch_size * patch_size;
    
    for (int r = half_size; r < image_h - half_size; r++) {
        for (int c = half_size; c < image_w - half_size; c++) {
            if (pyramid_idx == 0 && !boundary_mask(r, c)) continue;
            else if (pyramid_idx > 0 && !mask(r, c)) continue;

            // Find the 75th percentile distance (of those unmasked distances, if in initialization)
            vector<float> region_distances(patch_area, -1.f);
            vector<Vec2i> pixels(patch_area);

            ImageSliceCoords safe_region = patchRegion(Vec2i(r, c), image_h, image_w, true);

            unsigned int k = 0;
            for (int i = safe_region.row_start; i < safe_region.row_end; i++) {
                for (int j = safe_region.col_start; j < safe_region.col_end; j++) {
                    if (pyramid_idx > 0 || (pyramid_idx == 0 && !shrinking_mask(i, j))) {
                        pixels[k] = Vec2i(i, j);
                        region_distances[k] = distance_map(i, j);
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

            RGBPixel image_pixel;
            GradientPair texture_pixel;

            for (int i = 0; i < k; i++) {
                float entry_weight = scores[i] / scores_sum;

                Vec2i shift = shift_map(pixels[k].i, pixels[k].j);
                
                image_pixel += entry_weight * image(r + shift.i, c + shift.j);
                texture_pixel += entry_weight * texture(r + shift.i, c + shift.j);
            }

            image(r, c) = image_pixel;
            texture(r, c) = texture_pixel;
        }
    }
}

void PatchMatchInpainter::onionPeelInit()
{

}

PatchMatchInpainter::PatchMatchInpainter(unsigned int n_levels, unsigned int patch_size,
                                         image_t image, mask_t mask) 
{
    // Initialize all image, texture, etc. pyramids given the initial image and mask
    initPyramids(image, mask);

    // Initialize the level 0 shift map using random offsets for occluded pixels
    int last_level_index = n_levels - 1;
    int coarse_image_h = this->image_pyramid[last_level_index].height;
    int coarse_image_w = this->image_pyramid[last_level_index].width;

    for (int r = 0; r < coarse_image_h; r++) {
        for (int c = 0; c < coarse_image_h; c++) {
            Vec2i current_index = Vec2i(r, c);
            Vec2i candidate_index(current_index);

            while (this->dilated_mask_pyramid[last_level_index](candidate_index.i, candidate_index.j))
            {
                int random_row = rand() % (coarse_image_h - 2 * half_size) + half_size;
                int random_col = rand() % (coarse_image_w - 2 * half_size) + half_size;
                candidate_index = Vec2i(random_row, random_col);
            }

            this->shift_map_pyramid[last_level_index](r, c) = candidate_index - current_index;
        }
    }
}