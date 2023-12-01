#ifndef  __PATCH_MATCH_H__
#define  __PATCH_MATCH_H__

#include "utils.h"

typedef unsigned int* shift_map_t;
typedef float* distance_map_t;
typedef float* texture_t;
typedef bool* mask_t;
typedef int* image_t;

class PatchMatchInpainter {
private:
    unsigned int n_levels, patch_size, half_size;
    
    shift_map_t *shift_map_pyramid;
    distance_map_t *distance_map_pyramid;
    texture_t *texture_pyramid;
    mask_t *mask_pyramid;
    image_t *image_pyramid;
    
    /**
     * @brief Compute the image texture given the RGB image. The image texture is a 3D array where each element is defined as
     * the 2D vector of absolute values of the image gradients in (x, y) at that point
     * 
     * @param image Original image
     */
    void textureFromImage(image_t *image);

    /**
     * @brief Calculate the patch distance between patches A and B, each centered at a coordiante
     * 
     * @param centerA Center of patch A
     * @param centerB Center of patch B
     * @param masked If mask=true, apply the mask from patch A to both patches before calculating distance
     * @return float Patch distance metric from A to B
     */
    float patchDistance(Vec2i centerA, Vec2i centerB, bool masked=false);

    /**
     * @brief Initialize pyramid levels. Image pyramid for next highest level is the result of a Gaussian kernel
     * followed by a downsampling by some fraction rho=1/2. Mask pyramids are downsampled by rho with no kernels applied.
     * Texture pyramid at level L is the result of every (2^L)th pixel of the original texture. Distance map and shift map
     * levels are initialized to default values with correct sizes.
     * 
     * @param image Original image
     * @param mask Original mask
     */
    void initPyramids(image_t *image, mask_t *mask);
    
public:
    PatchMatchInpainter(unsigned int n_levels, unsigned int patch_size,
                        image_t *image, mask_t *mask);

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
    void approximateNearestNeighbor(unsigned int level, mask_t shrinking_mask=nullptr, mask_t boundary_mask=nullptr);

    /**
     * @brief Perform the image reconstruction step for the current level.
     * If level=0, perform image reconstruction for only those pixels in the boundary mask
     * Otherwise, perform image reconstruction for all pixels in the level's mask
     * 
     * @param level Current level in the process
     * @param shrinking_mask If level=0, the mask indicating the uninitialized portion of the hole
     * @param boundary_mask If level=0, the mask indicating pixels on the boundary of the uninitialized portion of the hole
     */
    void reconstructImage(unsigned int level, mask_t shrinking_mask=nullptr, mask_t boundary_mask=nullptr);

    /**
     * @brief Reconstruct the final image. This method does not use the weighted average of nearest neighbors of pixels
     * in each neighborhood. Instead, it uses the single pixel with the lowest NN distance value.
     */
    void reconstructFinalImage();

    /**
     * @brief Perform the inpainting procedure. First initializes the hole region using the onion-peel method, then
     * perform the iterative ANN search, reconstruction, and upsampling procedure
     */
    void inpaint();
};

#endif