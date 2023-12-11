#ifndef __PATCH_MATCH_H__
#define __PATCH_MATCH_H__

#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>

#include "cycle_timer.h"
#include "patch_match_utils.h"

using namespace std;
using namespace cv;

extern bool debug_mode_XXX, write_levels;

enum AlgorithmStage { INITIALIZATION = 0, NORMAL = 1, FINAL = 2 };

struct PatchMatchParams {
    unsigned int n_levels;
    unsigned int patch_size;
    unsigned int half_size;
    unsigned int patch_area;
    unsigned int n_iters;
    unsigned int n_iters_jfa;

    float lambda;

    // Default parameters
    PatchMatchParams() : n_levels(4), patch_size(5), n_iters(10), n_iters_jfa(1), lambda(5.f)
    {
        half_size = patch_size / 2;
        patch_area = patch_size * patch_size;
    }

    PatchMatchParams(unsigned int n_levels, unsigned int patch_size, unsigned int n_iters, unsigned int n_iters_jfa,
                     float lambda)
        : n_levels(n_levels), patch_size(patch_size), n_iters(n_iters), n_iters_jfa(n_iters_jfa), lambda(lambda)
    {
        if (patch_size < 0 || patch_size % 2 == 0) {
            fprintf(stderr, "Patch size must be odd positive integer (provided %d)\n", patch_size);
            exit(EXIT_FAILURE);
        }

        half_size = patch_size / 2;
        patch_area = patch_size * patch_size;
    }
};

struct TimingStats {
    double pyramid_build_time;
    double initialization_time;

    vector<double> level_times;
    vector<vector<double>> ann_times;
    vector<vector<double>> reconstruction_times;

    void prettyPrint()
    {
        double total_ann_time = 0;
        double total_reconstruction_time = 0;
        double total_time = 0;

        int n_levels = level_times.size();

        for (int i = 0; i < n_levels; ++i) {
            printf("Level %d times: \n", n_levels - 1 - i);

            double total_ann_time_ms = std::accumulate(ann_times[i].begin(), ann_times[i].end(), 0.f);
            double total_reconstruction_time_ms =
                std::accumulate(reconstruction_times[i].begin(), reconstruction_times[i].end(), 0.f);

            double avg_ann_time = total_ann_time_ms / static_cast<double>(ann_times[i].size());
            double avg_reconstruction_time =
                total_reconstruction_time_ms / static_cast<double>(reconstruction_times[i].size());
            double level_time = level_times[i];

            total_ann_time += avg_ann_time;
            total_reconstruction_time += avg_reconstruction_time;
            total_time += level_time;

            printf("  Level %d average ANN time:            %.2f ms\n", i, 1000.f * avg_ann_time);
            printf("  Level %d average reconstruction time: %.2f ms\n", i, 1000.f * avg_reconstruction_time);
            printf("  Level %d total time:                  %.2f ms\n", i, 1000.f * level_time);
        }

        printf("-----------------------------------\n");
        printf("Pyramid build time:                %.2f ms\n", 1000.f * pyramid_build_time);
        printf("Initialization time:               %.2f ms\n", 1000.f * initialization_time);
        printf("\n");
        printf("Total average ANN time:            %.2f ms\n", 1000.f * total_ann_time / n_levels);
        printf("Total average reconstruction time: %.2f ms\n", 1000.f * total_reconstruction_time / n_levels);
        printf("Total time:                        %.2f s\n", total_time);
    }
};

class PatchMatchInpainter {
   private:
    PatchMatchParams params;

    TimingStats timing_stats;

    shift_map_t *shift_map_pyramid;
    distance_map_t *distance_map_pyramid;
    texture_t *texture_pyramid;
    mask_t *mask_pyramid;
    mask_t *dilated_mask_pyramid;
    image_t *image_pyramid;
    Rect *hole_region_pyramid;

    Mat patch_dilation_element;
    Mat patch_size_zeros;
    Mat patch_size_ones;

    /**
     * @brief Create the bounding rectangle for a patch of size patch_size centered at a coordinate
     *
     * @param center Center coordinate
     * @param image_h Image height
     * @param image_w Image width
     * @param cutoff_padding If true, do not allow the patch to extend past/into the half_size padding
     * @return Rect The bounding rectangle for the patch
     */
    Rect patchRegion(Vec2i center, unsigned int image_h, unsigned int image_w, bool cutoff_padding = false)
    {
        int edge_size = cutoff_padding ? params.half_size : 0;

        Rect region =
            Rect(center[1] - params.half_size, center[0] - params.half_size, params.patch_size, params.patch_size);
        Rect image = Rect(edge_size, edge_size, image_w - 2 * edge_size, image_h - 2 * edge_size);

        return region & image;
    }

    /**
     * @brief Upsample a source matrix by a factor of 2 using nearest neighbor interpolation
     * and zero pad the border by padding amount
     *
     * @param src The source matrix to upsample
     * @param padding Amount of padding to add on all sides of the upsampled image
     * @param mul (Optional) If provided, multiply the upsampled image values by 2
     * @return Mat The upsampled and zero padded matrix
     */
    Mat upsampleZeroPad(const Mat &src, int padding, bool mul = false)
    {
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
     * centered at a coordinate. The distance calculation can be masked,
     * and the computation time is recorded.
     *
     * @param pyramid_idx Index of the current pyramid level
     * @param centerA Center of patch A
     * @param centerB Center of patch B
     * @param stage The current stage of the algorithm
     * @param init_shrinking_mask (Optional) If provided, apply the mask from patch A to both patches
     * before calculating distance
     * @return float Patch distance metric from A to B
     */
    float patchDistance(int pyramid_idx, Vec2i centerA, Vec2i centerB, AlgorithmStage stage, double &time, image_t& image, texture_t& texture,
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
    PatchMatchInpainter(image_t image, mask_t mask, PatchMatchParams params);

    ~PatchMatchInpainter();

    /**
     * @brief Perform the approximate nearest neighbor search for the current
     * level. This method uses the propagation and random search steps defined in the
     * PatchMatch paper to create the ANNF. The processing stage determines the
     * specific operations to be carried out during the search.
     *
     * @param pyramid_idx Index of the current pyramid level
     * @param stage The current stage of the algorithm
     * @param init_boundary_mask (Optional) If provided and stage=INIT, the mask indicating pixels on the
     * boundary of the uninitialized portion of the hole
     * @param init_shrinking_mask (Optional) If provided and stage=INIT, the mask indicating the uninitialized
     * portion of the hole
     */
    void approximateNearestNeighbor(int pyramid_idx, AlgorithmStage stage, double &patch_distance_time,
                                    optional<reference_wrapper<mask_t>> init_boundary_mask,
                                    optional<reference_wrapper<mask_t>> init_shrinking_mask);

    /**
     * @brief Perform the image reconstruction step for the current level.
     * The processing stage determines the specific operations to be carried out
     * during the reconstruction.
     *
     * @param pyramid_idx Index of the current pyramid level
     * @param stage The current stage of the algorithm
     * @param init_boundary_mask (Optional) If provided and stage=INIT, the mask indicating pixels on the
     * boundary of the uninitialized portion of the hole
     * @param init_shrinking_mask (Optional) If provided and stage=INIT, the mask indicating the uninitialized
     * portion of the hole
     */
    void reconstructImage(int pyramid_idx, AlgorithmStage stage, optional<reference_wrapper<mask_t>> init_boundary_mask,
                          optional<reference_wrapper<mask_t>> init_shrinking_mask);

    void annHelper(int r, int c, image_t& image, texture_t& texture, int pyramid_idx, vector<int>& jump_flood_radii, mask_t& dilated_mask,
                                        shift_map_t *active_shift_map, shift_map_t *prev_shift_map, shift_map_t& updated_shift_map, distance_map_t& updated_distance_map,
                                        double &patch_distance_time,  int& idx, int& jump_flood_radius, int radii_offsets[],
                                        optional<reference_wrapper<mask_t>> init_boundary_mask,
                                        optional<reference_wrapper<mask_t>> init_shrinking_mask);

    void reconstructionHelper(int r, int c, image_t& image, texture_t& texture, int pyramid_idx, mask_t mask, vector<int>& jump_flood_radii, mask_t& dilated_mask,
                                                   shift_map_t *active_shift_map, shift_map_t *prev_shift_map, shift_map_t& updated_shift_map, distance_map_t& updated_distance_map,
                                                   double &patch_distance_time, distance_map_t& distance_map, shift_map_t& shift_map,
                                                   image_t& updated_image, texture_t& updated_texture);
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