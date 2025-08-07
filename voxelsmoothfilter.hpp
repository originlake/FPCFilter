/* 
* voxelSmoothFilter.hpp
* This algorithm smooth and samples a point cloud using a voxel grid approach.
* In each voxel center, it will query neighbors within a given radius and
* compute the average normal of the points found. Then, it will compute
* the median magnitude along the average normal direction, and create a new point
* at the voxel center with this magnitude along the average normal direction.
* The resulting point cloud is both smoothed and sampled.
*/
#pragma once

#include <iostream>
#include <set>
#include <tuple>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include <omp.h>
#include <immintrin.h>

#include "ply.hpp"
#include "vendor/nanoflann.hpp"
#include "common.hpp"
#include "utils.hpp"


namespace FPCFilter {

    class VoxelSmoothFilter {
        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<
            double, PointCloud, double>, PointCloud, 3, std::size_t> KDTree;
        using Voxel = PointXYZ<int>;
        using Normal = PointXYZ<double>;
        using Coord = PointXYZ<double>;
        using IndexDist = nanoflann::ResultItem<size_t, double>;
        
        std::ostream& log;
        std::set<Voxel> voxels;

        const double E = 1e-3;
        double cell;
        double radius;
        double radiusSqr;
        double originX;
        double originY;
        double originZ;
        std::unique_ptr<KDTree> tree;
        const nanoflann::SearchParameters params;
        bool isVerbose;

    public:
        VoxelSmoothFilter(double radius, std::ostream& logstream, bool isVerbose) : originX(0), originY(0), originZ(0),
                isVerbose(isVerbose), log(logstream), radius(radius), cell(radius * std::sqrt(3.0)) {
            radiusSqr = radius * radius;
        }
        
        void run(PlyFile& file) {
            auto points = file.points;
            auto extras = file.extras;
            PointCloud pointCloud(points);

            const auto cnt = points.size();

            if (cnt == 0) 
                return;

            const auto hasNormals = file.hasNormals();

            if (!hasNormals) {
                log << " !> Error: input point cloud has no normals, cannot run VoxelSmoothFilter" << std::endl;
                throw std::invalid_argument("Input point cloud has no normals");
            }

            auto start = std::chrono::steady_clock::now();

            tree = std::make_unique<KDTree>(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(100, nanoflann::KDTreeSingleIndexAdaptorFlags::None, 0));
            tree->buildIndex();
            if (this->isVerbose) {
                const std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
                std::cout << " ?> Done building index in " << diff.count() << "s" << std::endl;
            }

            originX = points[0].x;
            originY = points[0].y;
            originZ = points[0].z;

            std::vector<PlyPoint> newPoints;
            newPoints.reserve(voxels.size());
            std::vector<PlyExtra> newExtras;
            newExtras.reserve(voxels.size());
            std::vector<PlyPoint> tmpPoints;
            tmpPoints.reserve(cnt / omp_get_max_threads());
            std::vector<PlyExtra> tmpExtras;
            tmpExtras.reserve(cnt / omp_get_max_threads());

            #pragma omp parallel private (tmpPoints, tmpExtras)
            {
                #pragma omp for
                for (auto i = 0; i < cnt; i++) {
                    const auto& point = points[i];
                    const auto& extra = extras[i];
                    Voxel v(0, 0, 0);
                    voxelize(point, v);
                    if (voxels.find(v) != voxels.end()) {
                        continue;
                    }
                    #pragma omp critical
                    {
                        voxels.insert(v);
                    }
                    PlyPoint newPoint(v.x * cell + originX, v.y * cell + originY, v.z * cell + originZ, 0, 0, 0, 0);
                    PlyExtra newExtra(0.0, 0.0, 0.0);
                    std::vector<IndexDist> indices;
                    radiusSearch(newPoint, indices);

                    if (indices.empty()) {
                        continue;
                    }

                    if (indices.size() == 1) {
                        // When there is only one point, use it directly
                        newPoint = points[indices[0].first];
                        newExtra = extras[indices[0].first];
                    } else if (indices[0].second < E) {
                        // When the first point is close enough to the voxel center, use it directly
                        newPoint = points[indices[0].first];
                        newExtra = extras[indices[0].first];
                    } else {
                        // Interpolate the center
                        computeCenterPoint(points, extras, indices, newPoint, newExtra);
                    }
                    tmpPoints.push_back(newPoint);
                    tmpExtras.push_back(newExtra);
                }

                #pragma omp critical
                {
                    if (this->isVerbose)
                        log << " ?> Sampled " << tmpPoints.size() << " points in thread " << omp_get_thread_num()
                            << std::endl;
                    newPoints.insert(newPoints.end(), tmpPoints.begin(), tmpPoints.end());
                    newExtras.insert(newExtras.end(), tmpExtras.begin(), tmpExtras.end());
                }
            }

            newPoints.shrink_to_fit();
            newExtras.shrink_to_fit();

            // Update the PlyFile with the new points and normals
            file.points = newPoints;
            file.extras = newExtras;
        }

    private:
        inline int fast_floor(double x)
        {
            int i = (int)x; /* truncate */
            return i - ( i > x ); /* convert trunc to floor */
        }

        void voxelize(const PlyPoint& point, Voxel& v) {
            double x = point.x;
            double y = point.y;
            double z = point.z;

            v.x = fast_floor((x - originX) / cell + 0.5);
            v.y = fast_floor((y - originY) / cell + 0.5);
            v.z = fast_floor((z - originZ) / cell + 0.5);
        }

        void radiusSearch(const PlyPoint& point, std::vector<IndexDist> &indices) const {
            indices.clear();
            std::array<double, 3> pt = {point.x, point.y, point.z};
            const size_t nMatches = tree->radiusSearch(&pt[0], radius, indices, this->params);
        }

        double inverseDistanceWeighting(double distance) const {
            return 1.0 / (1.0 + distance);
        }

        void computeCenterPoint(const std::vector<PlyPoint> &points, const std::vector<PlyExtra> &normals, const std::vector<IndexDist>& indices, PlyPoint& center, PlyExtra& centerNormal) const
        {
            const size_t n = indices.size();
            // Compute inverse distance weights, make sure to handle the case where the distance is zero
            std::vector<double> weights;
            double sumWeights = 0.0;
            weights.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                const auto& idx = indices[i];
                // Note that the distance is already squared in the radius search
                double w = inverseDistanceWeighting(idx.second);
                weights.push_back(w);
                sumWeights += w;
            }

            // Compute average normal
            double nx = 0.0, ny = 0.0, nz = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const auto& idx = indices[i];
                const auto& normal = normals[idx.first];
                nx += normal.nx * weights[i];
                ny += normal.ny * weights[i];
                nz += normal.nz * weights[i];
            }
            double normSqr = nx * nx + ny * ny + nz * nz;
            // Possibly points are all in opposite directions, so the norm of the vector can be closed to zero, use the closest point instead
            if (normSqr < 1e-9) {
                nx = normals[indices[0].first].nx;
                ny = normals[indices[0].first].ny;
                nz = normals[indices[0].first].nz;
            } else {
                double norm = std::sqrt(normSqr);
                nx /= norm;
                ny /= norm;
                nz /= norm;
            }

            // Compute weighted average of magnitudes along the normal direction
            double sumMagnitudes = 0.0;
            double sumRed = 0.0;
            double sumGreen = 0.0;
            double sumBlue = 0.0;
            double sumViews = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const auto& pt = points[indices[i].first];
                double dotProduct = (pt.x - center.x) * nx +
                                    (pt.y - center.y) * ny +
                                    (pt.z - center.z) * nz;
                sumMagnitudes += dotProduct * weights[i];
                sumRed += pt.red * weights[i];
                sumGreen += pt.green * weights[i];
                sumBlue += pt.blue * weights[i];
                sumViews += pt.views * weights[i];
            }

            double meanMagnitude = sumMagnitudes / sumWeights;
            // Assign the result to the center points
            center.x += nx * meanMagnitude;
            center.y += ny * meanMagnitude;
            center.z += nz * meanMagnitude;
            center.red = sumRed / sumWeights;
            center.green = sumGreen / sumWeights;
            center.blue = sumBlue / sumWeights;
            center.views = sumViews / sumWeights;
            centerNormal.nx = nx;
            centerNormal.ny = ny;
            centerNormal.nz = nz;
        }
    };
};