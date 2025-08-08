#pragma once

#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <random>
#include <memory>
#include <chrono>
#include <omp.h>

#include "../ply.hpp"
#include "nanoflann.hpp"
#include "../utils.hpp"

#define APPROX_HORIZON 0.98


namespace FPCFilter {

    class FastZSmoothFilter {
        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<
            double, PointCloud, double>, PointCloud, 3, std::size_t> KDTree;

        std::ostream& log;
        bool isVerbose;

        double radius;
        
        std::unique_ptr<KDTree> tree;

        const nanoflann::SearchParameters params;

    public:
        FastZSmoothFilter(double radius, std::ostream& logstream, bool isVerbose) : radius(radius), log(logstream), isVerbose(isVerbose), params(10) {
        }

        void radiusSearch(PlyPoint& point, double radius, std::vector<size_t> &indices) const {
            std::vector<nanoflann::ResultItem<size_t, double>> indices_dists;
            std::array<double, 3> pt = {point.x, point.y, point.z};
            const size_t nMatches = tree->radiusSearch(&pt[0], radius, indices_dists, this->params);
            indices.resize(nMatches);
            for(auto i = 0; i < nMatches; i++) {
                indices[i] = indices_dists[i].first;
            }
        }

        void run(PlyFile& file) {
            PointCloudWeighted pointCloud(file.points, 1.0, 1.0, 0.1);
            size_t np = pointCloud.pts.size();
            if (np == 0) {
                return;
            }

            const auto hasNormals = file.hasNormals();

            auto start = std::chrono::steady_clock::now();
        
            tree = std::make_unique<KDTree>(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(100, nanoflann::KDTreeSingleIndexAdaptorFlags::None, 0));
            tree->buildIndex();
            if (this->isVerbose) {
                const std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
                std::cout << " ?> Done building index in " << diff.count() << "s" << std::endl;
            }
            
            std::vector<double> newValuesZ(np);
            #pragma omp parallel for
            for (auto n =0; n < np; n++) {
                std::vector<size_t> indices;
                PlyPoint& point = file.points[n];
//                // only apply to points at near horizontal surfaces, approximated 11 degrees
//                if (file.extras[n].nz <= APPROX_HORIZON) {
//                    newValuesZ[n] = point.z;
//                    continue;
//                }
                radiusSearch(point, radius, indices);
                if (indices.empty()) {
                    newValuesZ[n] = point.z;
                    continue;
                }
                newValuesZ[n] = z_median(pointCloud, indices);
            }
            #pragma omp parallel for
            for (auto n =0; n < np; n++) {
                file.points[n].z = newValuesZ[n];
            }
        }

    private:
        double z_median(PointCloud &pointCloud, std::vector<size_t> &indices) const {
            std::vector<double> z_values;
            z_values.reserve(indices.size());
            for (auto i : indices) {
                z_values.push_back(pointCloud.pts[i].z);
            }
            if (z_values.size() % 2 == 0) {
                std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2, z_values.end());
                std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2 - 1, z_values.end());
                return (z_values[z_values.size() / 2] + z_values[z_values.size() / 2 - 1]) / 2;
            } else {
                std::nth_element(z_values.begin(), z_values.begin() + z_values.size() / 2, z_values.end());
                return z_values[z_values.size() / 2];
            }
        }

    };
    
}