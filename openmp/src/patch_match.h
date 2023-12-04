#ifndef  __PATCH_MATCH_H__
#define  __PATCH_MATCH_H__

#include <optional>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;
using namespace cv;

struct Dimension {
    unsigned int height;
    unsigned int width;
    unsigned int depth;
};

class PatchMatchInpainter {
private:
    int n_levels, patch_size, half_size;
    int curr_level;
    float lambda = 50;

    shift_map_t *shift_map_pyramid;
    distance_map_t *distance_map_pyramid;
    texture_t *texture_pyramid;
    mask_t *mask_pyramid;
    mask_t *dilated_mask_pyramid;
    image_t *image_pyramid;
    Dimension *dimensions_pyramid;

    mask_t initializationMask;
    mask_t initializationBoundary;
    
    Rect patchRegion(Vec2i center, unsigned int image_h, unsigned int image_w, bool cutoff_padding=false) {
        int edge_size = cutoff_padding ? half_size : 0;

        Rect region = Rect(center[1] - half_size, center[0] - half_size, patch_size, patch_size);
        Rect image = Rect(edge_size, edge_size, image_w - 2 * edge_size, image_h - 2 * edge_size);

        return region & image;
    }

    // ImageSliceCoords relativePatchRegion(cv2::Vec2i center, bool cutoff_padding=false) {
    //     auto level_height = this->dimensions_pyramid[curr_level].width, level_width = this->dimensions_pyramid[curr_level].height;
    //     size_t edge_size = cutoff_padding ? half_size : 0;

    //     ImageSliceCoords relative_region = patchRegion(center, cutoff_padding);
    //     relative_region.row_start -= center.i;
    //     relative_region.row_end -= center.i;
    //     relative_region.col_start -= center.j;
    //     relative_region.col_end -= center.j;

    //     return relative_region;
    // }

    /**
     * @brief Compute the image texture given the RGB image. The image texture is a 3D array where each element is defined as
     * the 2D vector of absolute values of the image gradients in (x, y) at that point
     * 
     * @param image Original image
     */
    void textureFromImage(image_t image);

    /**
     * @brief Calculate the patch distance between patches A and B, each centered at a coordiante
     * 
     * @param centerA Center of patch A
     * @param centerB Center of patch B
     * @param masked If mask=true, apply the mask from patch A to both patches before calculating distance
     * @return float Patch distance metric from A to B
     */
    float patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB, optional<reference_wrapper<mask_t>> init_shrinking_mask);

    /**
     * @brief Initialize pyramid levels. Image pyramid for next highest level is the result of a Gaussian kernel
     * followed by a downsampling by some fraction rho=1/2. Mask pyramids are downsampled by rho with no kernels applied.
     * Texture pyramid at level L is the result of every (2^L)th pixel of the original texture. Distance map and shift map
     * levels are initialized to default values with correct sizes.
     * 
     * @param image Original image
     * @param mask Original mask
     */
    void initPyramids(image_t image, mask_t mask);

    /**
     * @brief Get the current index into the image pyramid. We denote the initilization as level 0, meanwhile the rest
     * of the stages take place at level 1 to level n_levels, but a curr_level of 0 and 1 should return 0 in this function.
     * 
     * @return unsigned int An index 0 <= index < n_levels into the pyramids
     */
    unsigned int getPyramidIndex() {
        return max(curr_level - 1, 0);
    } 
    
public:
    PatchMatchInpainter(unsigned int n_levels, unsigned int patch_size,
                        image_t image, mask_t mask);

    ~PatchMatchInpainter();
    /**
     * @brief Perform the approximate nearest neighbor search for the current level. 
     * If level=0, perform approximate nearest neighbor search along only those pixels in the boundary mask
     * Otherwise, perform approximate nearest neighbor search for all pixels in the level's mask.
     * 
     * This method uses the propagation and random search steps defined in the PatchMatch paper to create the ANNF.
     * 
     * @param level Current level in the process
     * @param shrinking_mask If level=0, the mask indicating the uninitialized portion of the hole
     * @param boundary_mask If level=0, the mask indicating pixels on the boundary of the uninitialized portion of the hole
     */
    shift_map_t approximateNearestNeighbor(distance_map_t &distance_map);

    /**
     * @brief Perform the image reconstruction step for the current level.
     * If level=0, perform image reconstruction for only those pixels in the boundary mask
     * Otherwise, perform image reconstruction for all pixels in the level's mask
     * 
     * @param level Current level in the process
     * @param shrinking_mask If level=0, the mask indicating the uninitialized portion of the hole
     * @param boundary_mask If level=0, the mask indicating pixels on the boundary of the uninitialized portion of the hole
     */
    image_t reconstructImage(int pyramid_idx,
                             optional<reference_wrapper<mask_t>> init_boundary_mask,
                             optional<reference_wrapper<mask_t>> init_shrinking_mask);

    /**
     * @brief Reconstruct the final image. This method does not use the weighted average of nearest neighbors of pixels
     * in each neighborhood. Instead, it uses the single pixel with the lowest NN distance value.
     */
    void reconstructFinalImage();

    /**
     * @brief Perform onion-peel initialization of the image at the coarsest level of the image pyramid.
     */
    void onionPeelInit();

    /**
     * @brief Perform the inpainting procedure. First initializes the hole region using the onion-peel method, then
     * perform the iterative ANN search, reconstruction, and upsampling procedure
     */
    void inpaint();
};

#endif