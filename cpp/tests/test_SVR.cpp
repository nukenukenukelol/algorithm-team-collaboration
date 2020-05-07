#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "../spherical_volume_rendering_util.h"
#include <algorithm>

// Utilizes the Google Test suite.
// To run:
//       > cd cpp/tests/
//       >  git clone https://github.com/google/googletest.git
//       >  mkdir build && cd build && cmake .. && make all && ./test_SVR
//
// For information on Google Test, see: https://github.com/google/googletest/blob/master/googletest/README.md
// For examples of Google Test, see: https://github.com/google/googletest/tree/master/googletest/samples

namespace {

    // Determines equality amongst actual spherical voxels, and the expected spherical voxels.
    void expectEqualVoxels(const std::vector<svr::SphericalVoxel>& actual_voxels,
                           const std::vector<int>& expected_radial_voxels,
                           const std::vector<int>& expected_theta_voxels,
                           const std::vector<int>& expected_phi_voxels) {
        const std::size_t num_voxels = actual_voxels.size();
        std::vector<int> radial_voxels(num_voxels);
        std::vector<int> theta_voxels(num_voxels);
        std::vector<int> phi_voxels(num_voxels);
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), radial_voxels.begin(),
                       [](const svr::SphericalVoxel& sv) -> int { return sv.radial_voxel; });
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), theta_voxels.begin(),
                       [](const svr::SphericalVoxel& sv) -> int { return sv.angular_voxel; });
        std::transform(actual_voxels.cbegin(), actual_voxels.cend(), phi_voxels.begin(),
                       [](const svr::SphericalVoxel& sv) -> int { return sv.azimuthal_voxel; });
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
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(3.0, 3.0, 3.0);
        const FreeVec3 ray_direction(-2.0, -1.3, 1.0);
        const Ray ray(ray_origin, ray_direction);

        const double t_begin = 0.0;
        const double t_end = 15.0;
        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        EXPECT_EQ(actual_voxels.size(), 0);
    }

    TEST(SphericalCoordinateTraversal, RayBeginsWithinSphere) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-3.0, 4.0, 5.0);
        const FreeVec3 ray_direction(1.0, -1.0, -1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {2,3,4,4,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {1,1,1,0,3,3,3,3,3};
        const std::vector<int> expected_phi_voxels = {1,1,1,0,0,3,3,3,3};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayBeginsWithinSphereAndTimeBeginIsNotZero) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-3.0, 4.0, 5.0);
        const FreeVec3 ray_direction(1.0, -1.0, -1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 5.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {4,3,2,1};
        const std::vector<int> expected_theta_voxels = {3,3,3,3};
        const std::vector<int> expected_phi_voxels = {0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayEndsWithinSphere) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(13.0, -15.0, 16.0);
        const FreeVec3 ray_direction(-1.5, 1.2, -1.5);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 10.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 2, 3};
        const std::vector<int> expected_theta_voxels = {3, 3, 2, 2};
        const std::vector<int> expected_phi_voxels = {0, 0, 1, 1};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayBeginsAndEndsWithinSphere) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-3.0, 4.0, 5.0);
        const FreeVec3 ray_direction(1.0, -1.0, -1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 5.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {2,3,4,4,4};
        const std::vector<int> expected_theta_voxels = {1,1,1,0,3};
        const std::vector<int> expected_phi_voxels = {1,1,1,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, SphereCenteredAtOrigin) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, -13.0, -13.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {2,2,2,2,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {2,2,2,2,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RaySlightOffsetInXYPlane) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, -13.0, -13.0);
        const FreeVec3 ray_direction(1.0, 1.5, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 2, 3, 2, 2, 1};
        const std::vector<int> expected_theta_voxels = {2, 2, 1, 1, 1, 0, 0};
        const std::vector<int> expected_phi_voxels = {2, 2, 2, 2, 2, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }


    TEST(SphericalCoordinateTraversal, RayTravelsAlongXAxis) {
        const BoundVec3 min_bound(0.0, 0.0, 0.0);
        const BoundVec3 max_bound(30.0, 30.0, 30.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 8;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, 0.0, 0.0);
        const FreeVec3 ray_direction(1.0, 0.0, 0.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {3,3,3,3,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {1,1,1,1,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayTravelsAlongYAxis) {
        const BoundVec3 min_bound(0.0, 0.0, 0.0);
        const BoundVec3 max_bound(30.0, 30.0, 30.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 8;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(0.0, -15.0, 0.0);
        const FreeVec3 ray_direction(0.0, 1.0, 0.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {5,5,5,5,1,1,1,1};
        const std::vector<int> expected_phi_voxels = {0,0,0,0,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayTravelsAlongZAxis) {
        const BoundVec3 min_bound(0.0, 0.0, 0.0);
        const BoundVec3 max_bound(30.0, 30.0, 30.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 8;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(0.0, 0.0, -15.0);
        const FreeVec3 ray_direction(0.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {0,0,0,0,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {2,2,2,2,0,0,0,0};
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
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, 0.0);
        const FreeVec3 ray_direction(1.0, 1.0, 0.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {2,2,2,2,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {1,1,1,1,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayParallelToXZPlane) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, 0.0, -15.0);
        const FreeVec3 ray_direction(1.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {1,1,1,1,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {2,2,2,2,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayParallelToYZPlane) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(0.0, -15.0, -15.0);
        const FreeVec3 ray_direction(0.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1,2,3,4,4,3,2,1};
        const std::vector<int> expected_theta_voxels = {2,2,2,2,0,0,0,0};
        const std::vector<int> expected_phi_voxels = {2,2,2,2,0,0,0,0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayDirectionNegativeXPositiveYZ) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(13.0, -15.0, -15.0);
        const FreeVec3 ray_direction(-1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 3, 4, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {3, 3, 3, 2, 2, 1, 1, 1, 1};
        const std::vector<int> expected_phi_voxels = {3, 3, 3, 2, 2, 1, 1, 1, 1};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayDirectionNegativeYPositiveXZ) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, 17.0, -15.0);
        const FreeVec3 ray_direction(1.0, -1.2, 1.3);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 3, 4, 4, 3, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {1, 1, 1, 1, 1, 0, 0, 3, 3, 3};
        const std::vector<int> expected_phi_voxels = {2, 2, 2, 1, 1, 0, 0, 0, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayDirectionNegativeZPositiveXY) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-13.0, -12.0, 15.3);
        const FreeVec3 ray_direction(1.4, 2.0, -1.3);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 1, 2, 2, 1};
        const std::vector<int> expected_theta_voxels = {2, 1, 1, 0, 0};
        const std::vector<int> expected_phi_voxels = {1, 1, 1, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayDirectionNegativeXYZ) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(15.0, 12.0, 15.0);
        const FreeVec3 ray_direction(-1.4, -2.0, -1.3);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 1, 2, 1, 1};
        const std::vector<int> expected_theta_voxels = {0, 3, 3, 3, 2};
        const std::vector<int> expected_phi_voxels = {0, 0, 0, 0, 1};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, OddNumberAngularSections) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 9.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 3;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, -15.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.3);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 2, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {1, 1, 1, 1, 0, 0};
        const std::vector<int> expected_phi_voxels = {2, 2, 1, 1, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, OddNumberAzimuthalSections) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 3;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, -15.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 4, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {2, 2, 2, 2, 0, 0, 0, 0};
        const std::vector<int> expected_phi_voxels = {1, 1, 1, 1, 0, 0, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberOfRadialSectionsOne) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 40;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, -15.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                                         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 39, 38, 37,
                                                         36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
                                                         22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,
                                                         8, 7, 6, 5, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels =  {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0};
        const std::vector<int> expected_phi_voxels =    {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberRadialSectionsTwo) {
        const BoundVec3 min_bound(-200000.0, -200000.0, -200000.0);
        const BoundVec3 max_bound(200000.0, 200000.0, 200000.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10e6;
        const std::size_t num_radial_sections = 128;
        const std::size_t num_angular_sections = 1;
        const std::size_t num_azimuthal_sections = 1;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const double t_begin = 0.0;
        const double t_end = sphere_max_radius * 3;
        const BoundVec3 ray_origin(-421.875, -562.5, -(sphere_max_radius + 1.0));
        const FreeVec3 ray_direction(0.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        std::vector<int> expected_radial_voxels(num_radial_sections * 2 - 1); // { 1, 2, .. N-1, N, N-1, .. 3, 2, 1 }
        std::iota(expected_radial_voxels.begin(), expected_radial_voxels.begin() + num_radial_sections, 1);
        std::iota(expected_radial_voxels.rbegin(), expected_radial_voxels.rbegin() + num_radial_sections - 1, 1);

        std::vector<int> expected_theta_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_theta_voxels.begin(), expected_theta_voxels.end(), 1); // { 1, 1, ... 1, 1 }
        std::vector<int> expected_phi_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_phi_voxels.begin(), expected_phi_voxels.end(), 1);     // { 1, 1, ... 1, 1 }
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberRadialSectionsThree) {
        const BoundVec3 min_bound(-200000.0, -200000.0, -200000.0);
        const BoundVec3 max_bound(200000.0, 200000.0, 200000.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10e6;
        const std::size_t num_radial_sections = 256;
        const std::size_t num_angular_sections = 1;
        const std::size_t num_azimuthal_sections = 1;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const double t_begin = 0.0;
        const double t_end = sphere_max_radius * 3;
        const BoundVec3 ray_origin(-421.875, -562.5, -(sphere_max_radius + 1.0));
        const FreeVec3 ray_direction(0.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        std::vector<int> expected_radial_voxels(num_radial_sections * 2 - 1); // { 1, 2, ... N-1, N, N-1, ... 3, 2, 1 }
        std::iota(expected_radial_voxels.begin(), expected_radial_voxels.begin() + num_radial_sections, 1);
        std::iota(expected_radial_voxels.rbegin(), expected_radial_voxels.rbegin() + num_radial_sections - 1, 1);

        std::vector<int> expected_theta_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_theta_voxels.begin(), expected_theta_voxels.end(), 1); // { 1, 1, ..., 1, 1 }
        std::vector<int> expected_phi_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_phi_voxels.begin(), expected_phi_voxels.end(), 1);     // { 1, 1, ..., 1, 1 }
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberRadialSectionsFour) {
        const BoundVec3 min_bound(-200000.0, -200000.0, -200000.0);
        const BoundVec3 max_bound(200000.0, 200000.0, 200000.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10e6;
        const std::size_t num_radial_sections = 512;
        const std::size_t num_angular_sections = 1;
        const std::size_t num_azimuthal_sections = 1;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const double t_begin = 0.0;
        const double t_end = sphere_max_radius * 3;
        const BoundVec3 ray_origin(-421.875, -562.5, -(sphere_max_radius + 1.0));
        const FreeVec3 ray_direction(0.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        std::vector<int> expected_radial_voxels(num_radial_sections * 2 - 1); // { 1, 2, .. N-1, N, N-1, .. 3, 2, 1 }
        std::iota(expected_radial_voxels.begin(), expected_radial_voxels.begin() + num_radial_sections, 1);
        std::iota(expected_radial_voxels.rbegin(), expected_radial_voxels.rbegin() + num_radial_sections - 1, 1);

        std::vector<int> expected_theta_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_theta_voxels.begin(), expected_theta_voxels.end(), 1); // { 1, 1, ... 1, 1 }
        std::vector<int> expected_phi_voxels(num_radial_sections * 2 - 1);
        std::fill(expected_phi_voxels.begin(), expected_phi_voxels.end(), 1);     // { 1, 1, ... 1, 1 }
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberOfAngularSections) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 40;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, -15.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 4, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {24, 24, 24, 24, 4, 4, 4, 4};
        const std::vector<int> expected_phi_voxels = {2, 2, 2, 2, 0, 0, 0, 0};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, LargeNumberOfAzimuthalSections) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 40;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, -15.0, -15.0);
        const FreeVec3 ray_direction(1.0, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 4, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {2, 2, 2, 2, 0, 0, 0, 0};
        const std::vector<int> expected_phi_voxels = {24, 24, 24, 24, 4, 4, 4, 4};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, TimeBeginIsNotZero) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-15.0, 15.0, 15.0);
        const FreeVec3 ray_direction(1.0, -1.0, -1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.01;
        const double t_end = 50.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 4, 4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {1, 1, 1, 1, 3, 3, 3, 3};
        const std::vector<int> expected_phi_voxels = {1, 1, 1, 1, 3, 3, 3, 3};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayBeginsInOutermostRadiusAndEndsWithinSphere) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(-4.0, -4.0, -6.0);
        const FreeVec3 ray_direction(1.3, 1.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 4.3;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {1, 2, 3, 3, 4, 4};
        const std::vector<int> expected_theta_voxels = {2, 2, 2, 3, 3, 0};
        const std::vector<int> expected_phi_voxels = {2, 2, 2, 3, 3, 3};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(SphericalCoordinateTraversal, RayBeginsAtSphereOrigin) {
        const BoundVec3 min_bound(-20.0, -20.0, -20.0);
        const BoundVec3 max_bound(20.0, 20.0, 20.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10.0;
        const std::size_t num_radial_sections = 4;
        const std::size_t num_angular_sections = 4;
        const std::size_t num_azimuthal_sections = 4;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const BoundVec3 ray_origin(0.0, 0.0, 0.0);
        const FreeVec3 ray_direction(-1.5, 1.2, -1.5);
        const Ray ray(ray_origin, ray_direction);
        const double t_begin = 0.0;
        const double t_end = 30.0;

        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        const std::vector<int> expected_radial_voxels = {4, 3, 2, 1};
        const std::vector<int> expected_theta_voxels = {1, 1, 1, 1};
        const std::vector<int> expected_phi_voxels = {2, 2, 2, 2};
        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

    TEST(DISABLED_SphericalCoordinateTraversal, TangentialHitWithinSphere) {
        const BoundVec3 min_bound(-200000.0, -200000.0, -200000.0);
        const BoundVec3 max_bound(200000.0, 200000.0, 200000.0);
        const BoundVec3 sphere_center(0.0, 0.0, 0.0);
        const double sphere_max_radius = 10e3;
        const std::size_t num_radial_sections = 128;
        const std::size_t num_angular_sections = 1;
        const std::size_t num_azimuthal_sections = 1;
        const svr::SphericalVoxelGrid grid(min_bound, max_bound, num_radial_sections, num_angular_sections,
                                           num_azimuthal_sections, sphere_center, sphere_max_radius);
        const double t_begin = 0.0;
        const double t_end = sphere_max_radius * 3;
        const BoundVec3 ray_origin(-421.875, -562.5, -(sphere_max_radius + 1.0));
        const FreeVec3 ray_direction(0.0, 0.0, 1.0);
        const Ray ray(ray_origin, ray_direction);
        const auto actual_voxels = walkSphericalVolume(ray, grid, t_begin, t_end);
        // TODO: Fix expected values.
        std::vector<int> expected_radial_voxels(119 * 2 - 1);
        std::iota(expected_radial_voxels.begin(), expected_radial_voxels.begin() + 119, 1);
        std::iota(expected_radial_voxels.rbegin(), expected_radial_voxels.rbegin() + 119 - 1, 1);

        std::vector<int> expected_theta_voxels(119 * 2 - 1);
        std::vector<int> expected_phi_voxels(119 * 2 - 1);
        std::fill(expected_theta_voxels.begin(), expected_theta_voxels.end(), 1); // { 1, 1, ..., 1, 1 }
        std::fill(expected_phi_voxels.begin(), expected_phi_voxels.end(), 1);     // { 1, 1, ..., 1, 1 }

        expectEqualVoxels(actual_voxels, expected_radial_voxels, expected_theta_voxels, expected_phi_voxels);
    }

} // namespace