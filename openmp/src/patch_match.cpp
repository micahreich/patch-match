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

    texture_t image_texture = texture_t(image.height, image.width, 2);

    /**
    shift_map_t *shift_map_pyramid;
    distance_map_t *distance_map_pyramid;
    texture_t *texture_pyramid;
    mask_t *mask_pyramid;
    image_t *image_pyramid;*/
    image_pyramid[0] = image;
    mask_pyramid[0] = mask;
    texture_pyramid[0] = image_texture;
    
    for(unsigned int i = 1; i < n_levels; ++i) {
        image_t previous_image = image_pyramid[i-1];
        image_t next_level_image = applyGaussianFilter(previous_image, 3, 1.0);
        image_t next_level_image_downsampled = downsampleArray(next_level_image, 2);
        image_pyramid[i] = next_level_image_downsampled;

        int multiplier = pow(2, i);
        texture_t next_level_texture = downsampleArray(next_level_texture, multiplier);
        texture_pyramid[i] = next_level_texture;

        mask_t next_level_mask = downsampleArray(mask_pyramid[i-1], 2);
        mask_pyramid[i] = next_level_mask;
        
    }

    

    /**
    padding_width = [(HALF_SIZE, HALF_SIZE), (HALF_SIZE, HALF_SIZE), (0, 0)]

    # To prevent out of bounds errors, pad the image/texture/map at all levels with half the patch size
    # around the borders
    for i in range(0, n_pyramid_levels):
        image_pyramid[i] = np.pad(image_pyramid[i], pad_width=padding_width, mode='edge')
        texture_pyramid[i] = np.pad(texture_pyramid[i], pad_width=padding_width, mode='edge')
        mask_pyramid[i] = np.pad(mask_pyramid[i], pad_width=padding_width[:-1], mode='constant', constant_values=0)*/

    
}

float PatchMatchInpainter::patchDistance(Vec2i centerA, Vec2i centerB, bool masked=false)
{
    // Get the current level's image and texture pyramids
    int pyramid_idx = getPyramidIndex();

    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];

    size_t image_h = image.height, image_w = image.width;
    assert(inBounds(centerA.j, centerA.i, image_w, image_h, half_size)); // Should always be in bounds (outside padding)

    mask_t mask = this->mask_pyramid[pyramid_idx];
    float occluded_patch_area = patch_size * patch_size;
    
    // If masked, calculate how many pixels are unmasked in the region
    if (masked) {
        ImageSliceCoords regionA = patchRegion(centerA);
        occluded_patch_area = 0.f;

        for (size_t r = regionA.row_start; r < regionA.row_end; r++) {
            for (size_t c = regionA.col_start; r < regionA.col_end; c++) {
                occluded_patch_area += !mask(r, c);
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

            if (masked && mask(regionA_r, regionA_c)) continue;

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

image_t PatchMatchInpainter::reconstructImage()
{
    // Get the current level's image and texture pyramids
    int pyramid_idx = getPyramidIndex();

    image_t image = this->image_pyramid[pyramid_idx];
    texture_t texture = this->texture_pyramid[pyramid_idx];

    size_t image_h = image.height, image_w = image.width;
    

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