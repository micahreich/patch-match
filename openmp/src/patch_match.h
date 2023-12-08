#ifndef __PATCH_MATCH_H__
#define __PATCH_MATCH_H__

#include <opencv2/opencv.hpp>
#include <optional>

#include "patch_match_utils.h"

using namespace std;
using namespace cv;

extern bool debug_mode;

enum AlgorithmStage { INITIALIZATION = 0, NORMAL = 1, FINAL = 2 };

struct PatchMatchParams {
    unsigned int n_levels;
    unsigned int patch_size;
    unsigned int half_size;
    unsigned int n_iters;
    unsigned int n_iters_init;

    float lambda;

    // Default parameters
    PatchMatchParams() : n_levels(4), patch_size(5), n_iters(10), n_iters_init(1), lambda(5.f)
    {
        half_size = patch_size / 2;
    }

    PatchMatchParams(unsigned int n_levels, unsigned int patch_size, unsigned int n_iters, unsigned int n_iters_init,
                     float lambda)
        : n_levels(n_levels), patch_size(patch_size), n_iters(n_iters), n_iters_init(n_iters_init), lambda(lambda)
    {
        half_size = patch_size / 2;
    }
};

class PatchMatchInpainter {
   private:
    PatchMatchParams params;

    shift_map_t *shift_map_pyramid;
    distance_map_t *distance_map_pyramid;
    texture_t *texture_pyramid;
    mask_t *mask_pyramid;
    mask_t *dilated_mask_pyramid;
    image_t *image_pyramid;

    Mat patch_dilation_element;

    Rect patchRegion(Vec2i center, unsigned int image_h, unsigned int image_w, bool cutoff_padding = false)
    {
        int edge_size = cutoff_padding ? params.half_size : 0;

        Rect region =
            Rect(center[1] - params.half_size, center[0] - params.half_size, params.patch_size, params.patch_size);
        Rect image = Rect(edge_size, edge_size, image_w - 2 * edge_size, image_h - 2 * edge_size);

        return region & image;
    }

    template <typename T>
    Mat upsampleZeroPad(const Mat &src, int padding, bool mul = false)
    {
        // int image_h = src.rows, image_w = src.cols;
        // int new_image_h = 2 * (image_h - 2 * params.half_size) + 2 * params.half_size;
        // int new_image_w = 2 * (image_w - 2 * params.half_size) + 2 * params.half_size;

        // int multiplier = mul ? 2 : 1;

        // Vec2i half_size_vec = Vec2i(params.half_size, params.half_size);
        // Mat upsampled_src = Mat::zeros(new_image_h, new_image_w, src.type());

        // for (int r = params.half_size; r < new_image_h - params.half_size; r++) {
        //     for (int c = params.half_size; c < new_image_w - params.half_size; c++) {
        //         Vec2i current_coord = Vec2i(r, c);
        //         Vec2i translated_coord = current_coord - half_size_vec;
        //         Vec2i downsampled_coord = (translated_coord / 2) + half_size_vec;

        //         upsampled_src.at<T>(current_coord[0], current_coord[1]) = multiplier *
        //         src.at<T>(downsampled_coord[0], downsampled_coord[1]);
        //     }
        // }

        // return upsampled_src;

        Rect inner_region = Rect(padding, padding, src.cols - 2 * padding, src.rows - 2 * padding);
        Mat inner = src(inner_region);

        Mat upsampled_src;
        resize(inner, upsampled_src, Size(), 2, 2, INTER_NEAREST);

        if (mul) upsampled_src *= 2;

        Mat padded;
        copyMakeBorder(upsampled_src, padded, padding, padding, padding, padding, BORDER_CONSTANT, Scalar::all(0));

        return padded;
    }

    /**
     * @brief Calculate the patch distance between patches A and B, each
     * centered at a coordiante
     *
     * @param centerA Center of patch A
     * @param centerB Center of patch B
     * @param masked If mask=true, apply the mask from patch A to both patches
     * before calculating distance
     * @return float Patch distance metric from A to B
     */
    float patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB, AlgorithmStage stage,
                        optional<reference_wrapper<mask_t>> init_shrinking_mask, string marker);

    /**
     * @brief Initialize pyramid levels. Image pyramid for next highest level is
     * the result of a Gaussian kernel followed by a downsampling by some
     * fraction rho=1/2. Mask pyramids are downsampled by rho with no kernels
     * applied. Texture pyramid at level L is the result of every (2^L)th pixel
     * of the original texture. Distance map and shift map levels are
     * initialized to default values with correct sizes.
     *
     * @param image Original image
     * @param mask Original mask
     */
    void initPyramids(image_t image, mask_t mask);

   public:
    PatchMatchInpainter(PatchMatchParams params, image_t image, mask_t mask);

    ~PatchMatchInpainter();
    /**
     * @brief Perform the approximate nearest neighbor search for the current
     * level. If level=0, perform approximate nearest neighbor search along only
     * those pixels in the boundary mask Otherwise, perform approximate nearest
     * neighbor search for all pixels in the level's mask.
     *
     * This method uses the propagation and random search steps defined in the
     * PatchMatch paper to create the ANNF.
     *
     * @param level Current level in the process
     * @param shrinking_mask If level=0, the mask indicating the uninitialized
     * portion of the hole
     * @param boundary_mask If level=0, the mask indicating pixels on the
     * boundary of the uninitialized portion of the hole
     */
    void approximateNearestNeighbor(int pyramid_idx, AlgorithmStage stage,
                                    optional<reference_wrapper<mask_t>> init_boundary_mask,
                                    optional<reference_wrapper<mask_t>> init_shrinking_mask);

    /**
     * @brief Perform the image reconstruction step for the current level.
     * If level=0, perform image reconstruction for only those pixels in the
     * boundary mask Otherwise, perform image reconstruction for all pixels in
     * the level's mask
     *
     * @param level Current level in the process
     * @param shrinking_mask If level=0, the mask indicating the uninitialized
     * portion of the hole
     * @param boundary_mask If level=0, the mask indicating pixels on the
     * boundary of the uninitialized portion of the hole
     */
    void reconstructImage(int pyramid_idx, AlgorithmStage stage, optional<reference_wrapper<mask_t>> init_boundary_mask,
                          optional<reference_wrapper<mask_t>> init_shrinking_mask);

    /**
     * @brief Perform onion-peel initialization of the image at the coarsest
     * level of the image pyramid.
     */
    void onionPeelInit();

    /**
     * @brief Perform the inpainting procedure. First initializes the hole
     * region using the onion-peel method, then perform the iterative ANN
     * search, reconstruction, and upsampling procedure
     */
    image_t inpaint();
};

#endif