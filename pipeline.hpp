
#pragma once

#include <iostream>
#include <functional>
#include <filesystem>
#include <map>

#include <fstream>
#include <sstream>
#include <string>

#include "ply.hpp"
#include "common.hpp"

#include "filters/fastsamplefilter.hpp"
#include "filters/fastoutlierfilter.hpp"
#include "filters/fastzsmoothfilter.hpp"
#include "filters/voxelsamplefilter.hpp"
#include "filters/faststats.h"

namespace fs = std::filesystem;

namespace FPCFilter
{

	class Pipeline
	{

		std::unique_ptr<PlyFile> ply;

		std::ostream& log;

		std::string source;
		bool isLoaded = false;
		bool isVerbose = false;
		nlohmann::json *stats;

	public:
		Pipeline(const std::string &source, std::ostream& logstream, const bool verbose, nlohmann::json *stats) : 
			source(source), isVerbose(verbose), log(logstream), stats(stats) {}

		void load()
		{
			const auto start = std::chrono::steady_clock::now();

			this->ply = std::make_unique<PlyFile>(this->source);

			this->isLoaded = true;

			if (this->isVerbose) {
				const std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
				log << " ?> Loaded " << this->ply->points.size() << " points in " << diff.count() << "s" << std::endl;
			}

			FastStats fastStats(log, isVerbose);
			fastStats.run(*this->ply);
		}

		void crop(const Polygon &p)
		{

			if (!this->isLoaded)
			{
				const auto start = std::chrono::steady_clock::now();

				this->ply = std::make_unique<PlyFile>(this->source, [&p](const float x, const float y, const float z)
													  { return p.inside(x, y); });

				this->isLoaded = true;

				if (this->isVerbose) {
					const std::chrono::duration<double> diff = std::chrono::steady_clock::now() - start;
					log << " ?> Loaded " << this->ply->points.size() << " points (cropped) in " << diff.count() << "s" << std::endl;
				}

				FastStats fastStats(log, isVerbose);
				fastStats.run(*this->ply);

				return;
			}

			throw NotImplementedException("Filtering loaded ply is not supported");
		}

		void sample(double radius)
		{
			if (!this->isLoaded)
				this->load();

//			FastSampleFilter filter(radius, this->log, this->isVerbose);
			VoxelSampleFilter filter(radius, this->log, this->isVerbose);

			filter.run(*this->ply);
		}

		void filter(double std, int meank)
		{
			if (!this->isLoaded)
				this->load();

			FastOutlierFilter filter(std, meank, this->log, this->isVerbose);

			filter.run(*this->ply);
		}

		void smooth(double smooth_factor)
		{
			if (!this->isLoaded)
				this->load();

			FastZSmoothFilter filter(this->ply->spacing * 4, this->log, this->isVerbose);
			filter.run(*this->ply);
		}

		void runStats()
		{
			if (!this->isLoaded)
				this->load();
			FastStats fastStats(log, isVerbose);
			fastStats.run(*this->ply);
			(*stats)["spacing"] = this->ply->spacing;
		}

		void write(const std::string &target)
		{

			if (!this->isLoaded)
				this->load();

			if (fs::exists(target))
				fs::remove(target);

			std::ofstream writer(target, std::ofstream::binary);

			if (!writer.is_open())
				throw std::invalid_argument(std::string("Cannot open file ") + target);

			this->ply->write(writer);

			writer.close();
		};
	};

}