#pragma once

#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <omp.h>

#include "../ply.hpp"
#include "../common.hpp"

namespace FPCFilter {

    class FastSampleFilter {
        using Voxel = PointXYZ<int>;
        using Coord = PointXYZ<double>;
        using CoordList = std::vector<Coord>;

        std::ostream& log;

        std::map<Voxel, CoordList> voxels;

        double cell;
        double radius;
        double radiusSqr;
        double originX;
        double originY;
        double originZ;

        bool isVerbose;

    public:
        FastSampleFilter(double radius, std::ostream& logstream, bool isVerbose) : originX(0), originY(0), originZ(0), 
                isVerbose(isVerbose), log(logstream), radius(radius), radiusSqr(radius * radius) {
            cell = 2.0 * radius / std::sqrt(3.0);
        }

        void run(PlyFile& file) {
            
            auto points = file.points;
            auto extras = file.extras;
            
            const auto cnt = points.size();

            if (cnt == 0) 
                return;
            
            const auto hasNormals = file.hasNormals();

            originX = points[0].x;
            originY = points[0].y;
            originZ = points[0].z;

            if (hasNormals) {

                std::vector<PlyPoint> newPoints;
                newPoints.reserve(cnt);

                std::vector<PlyExtra> newExtras;
                newExtras.reserve(cnt);

                std::vector<PlyPoint> tmpPoints;
                tmpPoints.reserve(cnt / omp_get_max_threads());
                std::vector<PlyExtra> tmpExtras;
                tmpExtras.reserve(cnt / omp_get_max_threads());

                #pragma omp parallel private (tmpPoints, tmpExtras)
                {
                    #pragma omp for
                    for (auto n = 0; n < cnt; n++) {

                        const auto& point = points[n];
                        const auto& extra = extras[n];

                        if (this->safe_voxelize(point)) {
                            tmpPoints.push_back(point);
                            tmpExtras.push_back(extra);
                        }
                    }

                    #pragma omp critical
                    {
                        if (this->isVerbose)
                            log << " ?> Sampled " << tmpPoints.size() << " points in thread " << omp_get_thread_num() << std::endl;

                        newPoints.insert(newPoints.end(), tmpPoints.begin(), tmpPoints.end());
                        newExtras.insert(newExtras.end(), tmpExtras.begin(), tmpExtras.end());
                    }
                }

                newPoints.shrink_to_fit();
                newExtras.shrink_to_fit();

                file.points = newPoints;
                file.extras = newExtras;

            } else {

                auto res = std::remove_if(points.begin(), points.end(), [this](PlyPoint& point) { return !this->safe_voxelize(point); });
                points.erase(res, points.end());
                
            }
        }

    private:

        inline int fast_floor(double x)
        {
            int i = (int)x; /* truncate */
            return i - ( i > x ); /* convert trunc to floor */
        }

        bool safe_voxelize(const PlyPoint& point) {

            double x = point.x;
            double y = point.y;
            double z = point.z;

            // Get voxel indices for current point.
            auto v = Voxel(fast_floor((x - originX) / cell), 
                           fast_floor((y - originY) / cell), 
                           fast_floor((z - originZ) / cell));

            // Check current voxel before any of the neighbors. We will most often have
            // points that are too close in the point's enclosing voxel, thus saving
            // cycles.
            if (voxels.find(v) != voxels.end())
            {
                // Get list of coordinates in candidate voxel.
                CoordList coords = voxels[v];
                for (Coord const& coord : coords)
                {
                    // Compute Euclidean distance between current point and
                    // candidate voxel.
                    double xv = coord.x;
                    double yv = coord.y;
                    double zv = coord.z;
                    double distSqr =
                        (xv - x) * (xv - x) + (yv - y) * (yv - y) + (zv - z) * (zv - z);

                    // If any point is closer than the minimum radius, we can
                    // immediately return false, as the minimum distance
                    // criterion is violated.
                    if (distSqr < radiusSqr)
                        return false;
                }
            }

            // Iterate over immediate neighbors of current voxel, computing minimum
            // distance between any already added point and the current point.
            for (int xi = v.x - 1; xi < v.x + 2; ++xi)
            {
                for (int yi = v.y - 1; yi < v.y + 2; ++yi)
                {
                    for (int zi = v.z - 1; zi < v.z + 2; ++zi)
                    {
                        auto candidate = Voxel(xi, yi, zi);

                        // We have already visited the center voxel, and can skip it.
                        if (v == candidate)
                            continue;

                        // Check that candidate voxel is occupied.
                        if (voxels.find(candidate) == voxels.end())
                            continue;

                        // Get list of coordinates in candidate voxel.
                        CoordList coords = voxels[candidate];
                        for (Coord const& coord : coords)
                        {
                            // Compute Euclidean distance between current point and
                            // candidate voxel.
                            double xv = coord.x;
                            double yv = coord.y;
                            double zv = coord.z;
                            double distSqr = (xv - x) * (xv - x) + (yv - y) * (yv - y) +
                                            (zv - z) * (zv - z);

                            // If any point is closer than the minimum radius, we can
                            // immediately return false, as the minimum distance
                            // criterion is violated.
                            if (distSqr < radiusSqr)
                                return false;
                        }
                    }
                }
            }

            auto coord = Coord(x, y, z);

            #pragma omp critical
            {
                if (voxels.find(v) != voxels.end())
                {
                    voxels[v].push_back(coord);
                }
                else
                {
                    CoordList coords;
                    coords.push_back(coord);
                    voxels.emplace(v, coords);
                }
            }

            return true;
        }
        
    };

}