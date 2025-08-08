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

#include "../ply.hpp"
#include "nanoflann.hpp"
#include "../common.hpp"
#include "../utils.hpp"

namespace FPCFilter {
    class FastStats {
        typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<
                double, PointCloud, double>, PointCloud, 3, std::size_t> KDTree;
        std::ostream& log;
        bool isVerbose;
        std::unique_ptr<KDTree> tree;
        const nanoflann::SearchParameters params;

    public:
        FastStats(std::ostream& logstream, bool isVerbose) : log(logstream), isVerbose(isVerbose), params(nanoflann::SearchParameters()) {}

        void run(PlyFile& file) {
            PointCloud pointCloud(file.points);

            auto start = std::chrono::steady_clock::now();

            tree = std::make_unique<KDTree>(3, pointCloud, nanoflann::KDTreeSingleIndexAdaptorParams(100, nanoflann::KDTreeSingleIndexAdaptorFlags::None, 0));
            tree->buildIndex();

            if (this->isVerbose) {
                const std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
                std::cout << " ?> Done building index in " << diff.count() << "s" << std::endl;
            }

            size_t np = file.points.size();

            // Compute neighbor median distance over closest neighbors and bounding box
            std::vector<size_t> indices;
            std::vector<double> sqr_dists;
            size_t SAMPLES = std::min<size_t>(np, 10000);

            size_t count = 3;
            std::vector<double> distances(SAMPLES, 0.0);
            std::unordered_map<uint64_t, size_t> dist_map;
            std::vector<double> all_distances;
            std::random_device rd;
            std::mt19937_64 gen(rd());
            std::uniform_int_distribution<size_t> randomDis(
                    0, np - 1
            );

            #pragma omp parallel private (indices, sqr_dists)
            {
                indices.resize(count);
                sqr_dists.resize(count);

                #pragma omp for
                for (long long i = 0; i < SAMPLES; ++i)
                {
                    const size_t idx = randomDis(gen);
                    knnSearch(file.points[idx], count, indices, sqr_dists);

                    double sum = 0.0;
                    for (size_t j = 1; j < count; ++j)
                    {
                        sum += std::sqrt(sqr_dists[j]);
                    }
                    sum /= count;

                    #pragma omp critical
                    {
                        uint64_t k = std::ceil(sum * 100);
                        if (dist_map.find(k) == dist_map.end()){
                            dist_map[k] = 1;
                        }else{
                            dist_map[k] += 1;
                        }
                    }
                    indices.clear(); indices.resize(count);
                    sqr_dists.clear(); sqr_dists.resize(count);
                }
            }

            uint64_t max_val = std::numeric_limits<uint64_t>::min();
            int d = 0;
            for (auto it : dist_map){
                if (it.second > max_val){
                    d = it.first;
                    max_val = it.second;
                }
            }

            file.spacing = static_cast<double>(d) / 100.0;
            auto bbox = tree->root_bbox_;
            file.minX = bbox[0].low;
            file.maxX = bbox[0].high;
            file.minY = bbox[1].low;
            file.maxY = bbox[1].high;
            file.minZ = bbox[2].low;
            file.maxZ = bbox[2].high;
            std::cout << " -> Spacing estimation completed (" << file.spacing << ")" << std::endl;
            std::cout << " -> Bounding box (" << file.minX << ", " << file.maxX << ", " << file.minY << ", " << file.maxY << ", " << file.minZ << ", " << file.maxZ << ")" << std::endl;
        }
    private:
        void knnSearch(PlyPoint& point, size_t k,
                       std::vector<size_t>& indices, std::vector<double>& sqr_dists) const
        {
            nanoflann::KNNResultSet<double, size_t, size_t> resultSet(k);

            resultSet.init(&indices.front(), &sqr_dists.front());

            std::array<double, 3> pt = {point.x, point.y, point.z};
            tree->findNeighbors(resultSet, &pt[0], this->params);

        }
    };
}