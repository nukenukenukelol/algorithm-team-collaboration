#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "../spherical_volume_rendering_util.h"

// Utilizes the Google Test suite.
// To run, in the ../cpp/testing directory:
//   1. Clone the googletest repository.
//   >  git clone https://github.com/google/googletest.git
//
//   2. CMake
//   >  mkdir build && cd build && cmake .. && make all && ./run_svr_tests
//
// For information on Google Test, see: https://github.com/google/googletest/blob/master/googletest/README.md
// For examples of Google Test, see: https://github.com/google/googletest/tree/master/googletest/samples

namespace {
    // Determines equality amongst actual spherical voxels, and the expected spherical voxels.
    void expectEqualVoxels(const std::vector<SphericalVoxel>& actual_voxels,
                           const std::vector<std::size_t>& expected_radial_voxels,
                           const std::vector<std::size_t>& expected_theta_voxels,
                           const std::vector<std::size_t>& expected_phi_voxels) {
        const std::size_t num_voxels = actual_voxels.size();
        std::vector<std::size_t> radial_voxels(num_voxels);
        std::vector<std::size_t> theta_voxels(num_voxels);
        std::vector<std::size_t> phi_voxels(num_voxels);
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), radial_voxels.begin(),
                [](const SphericalVoxel& sv) -> std::size_t { return sv.radial_voxel; });
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), theta_voxels.begin(),
                       [](const SphericalVoxel& sv) -> std::size_t { return sv.angular_voxel; });
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), phi_voxels.begin(),
                       [](const SphericalVoxel& sv) -> std::size_t { return sv.azimuthal_voxel; });
        EXPECT_THAT(radial_voxels, testing::ContainerEq(expected_radial_voxels));
        EXPECT_THAT(theta_voxels, testing::ContainerEq(expected_theta_voxels));
        EXPECT_THAT(phi_voxels, testing::ContainerEq(expected_phi_voxels));
    }

    TEST(SphericalCoordinateTraversal, RayDoesNotEnterSphere) {
        const BoundVec3 min_bound(0.0, 0.0, 0.0);
        const BoundVec3 max_bound(30.0, 30.0, 30.0);
        const BoundVec3 sphere_center(15.0, 15.0, 15.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 8;
        const std::size_t num_azimuthal_sections = 4;
        const SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections,
                                      num_angular_sections,
                                      num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(3.0, 3.0, 3.0);
        const FreeVec3 ray_direction(-2.0, -1.3, 1.0);
        const Ray ray(ray_origin, ray_direction);

        const double t_begin = 0.0;
        const double t_end = 15.0;
        const auto actual_voxels = sphericalCoordinateVoxelTraversal(ray, grid, t_begin, t_end);
        EXPECT_EQ(actual_voxels.size(), 0);
    }

    TEST(SphericalCoordinateTraversal, SphereCenteredAtOrigin) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections,
                                      num_angular_sections,
                                      num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, -13.0, -13.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = sphericalCoordinateVoxelTraversal(ray, grid, t_begin, t_end);
        const std::vector<std::size_t> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<std::size_t> expected_theta_voxels = {2,2,2,2,0,0,0,0};
        const std::vector<std::size_t> expected_phi_voxels = {2,2,2,2,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayDirectionSlightlyOffsetInXYPlane) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections,
                                      num_angular_sections,
                                      num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, -13.0, -13.0);
        const FreeVec3 ray_direction(1.0, 1.5, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = sphericalCoordinateVoxelTraversal(ray, grid, t_begin, t_end);
        const std::vector<std::size_t> expected_radial_voxels = {1,2,2,3,2,2,2,1};
        const std::vector<std::size_t> expected_theta_voxels = {2,2,1,1,1,1,0,0};
        const std::vector<std::size_t> expected_phi_voxels = {2,2,2,2,2,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayParallelToXYPlane) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections,
                                      num_angular_sections,
                                      num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, 0.0);
        const FreeVec3 ray_direction(1.0, 1.0, 0.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = sphericalCoordinateVoxelTraversal(ray, grid, t_begin, t_end);
        const std::vector<std::size_t> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<std::size_t> expected_theta_voxels = {2,2,2,2,0,0,0,0};
        const std::vector<std::size_t> expected_phi_voxels = {2,2,2,2,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }
}