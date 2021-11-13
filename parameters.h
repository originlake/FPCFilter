
#include <iostream>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include "common.h"
#include "vendor/cxxopts.hpp"
#include "utils.h"
#include "vendor/json.hpp"

#define DEFAULT_STD_DEV "2.5"
#define DEFAULT_SAMPLE_RADIUS "0"

#ifdef DEBUG
#define DEFAULT_VERBOSE "true"
#else
#define DEFAULT_VERBOSE "false"
#endif

namespace fs = std::filesystem;

namespace FPCFilter
{

	class Parameters
	{

		std::optional<Polygon> extractPolygon(const std::string& boundary)
		{

			Polygon polygon;

			std::ifstream i(boundary);
			nlohmann::json j;
			i >> j;

			const auto features = j["features"];

			if (features.empty())
				return std::nullopt;


			for (const auto& f : features)
			{
				const auto geometry = f["geometry"];

				if (geometry["type"] != "Polygon")
					continue;

				const auto coordinates = geometry["coordinates"][0];

				// Add the points
				for (auto& coord : coordinates)
					polygon.addPoint(coord[0], coord[1]);

				return polygon;
			}

			return std::nullopt;
		}

	public:
		std::string input;
		std::string output;
		std::optional<Polygon> boundary;
		double std;
		double radius;
		int concurrency;
		bool verbose;

		Parameters(const int argc, char** argv)
		{

			cxxopts::Options options("FPCFilter", "Fast Point Cloud Filtering");

			options.show_positional_help();

			options.add_options()
				("i,input", "Input point cloud", cxxopts::value<std::string>())
				("o,output", "Output point cloud", cxxopts::value<std::string>())
				("b,boundary", "Process boundary (GeoJSON POLYGON)", cxxopts::value<std::string>()->default_value(""))
				("s,std", "Standard deviation", cxxopts::value<double>()->default_value(DEFAULT_STD_DEV))
				("r,radius", "Sample radius", cxxopts::value<double>()->default_value(DEFAULT_SAMPLE_RADIUS))
				("c,concurrency", "Max concurrency", cxxopts::value<int>()->default_value("0"))
				("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value(DEFAULT_VERBOSE));

			options.parse_positional({ "input", "output" });

			const auto result = options.parse(argc, argv);

			if (!result.count("input") || !result.count("output"))
				throw std::invalid_argument(options.help());

			input = result["input"].as<std::string>();

			if (input.empty())
				throw std::invalid_argument("Input file is empty");

			if (!fs::exists(input))
				throw std::invalid_argument(string_format("Input file '{}' does not exist", input));

			output = result["output"].as<std::string>();

			if (output.empty())
				throw std::invalid_argument("Output file is empty");

			std = result["std"].as<double>();

			if (std < 0)
				throw std::invalid_argument("Standard deviation cannot be less than 0");

			radius = result["radius"].as<double>();

			if (radius < 0)
				throw std::invalid_argument("Radius cannot be less than 0");

			concurrency = result["concurrency"].as<int>();

			if (concurrency == 0)
				concurrency = std::max(omp_get_num_procs(), 1);
			else if (concurrency < 0)
				throw std::invalid_argument("Concurrency cannot be less than 0");

			verbose = result["verbose"].as<bool>();

			const auto boundaryFile = result["boundary"].as<std::string>();

			if (!boundaryFile.empty())
			{

				boundary = extractPolygon(boundaryFile);

				if (!boundary.has_value())
					throw std::invalid_argument(string_format("Boundary file '{}' does not contain a valid GeoJSON POLYGON", boundaryFile));

			}

		}
	};

}