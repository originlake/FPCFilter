	
    #pragma once

    #include <iostream>
    #include <filesystem>

    namespace FPCFilter {
    
    class PlyPoint {
	public:
		float x;
		float y;
		float z;

		uint8_t red;
		uint8_t blue;
		uint8_t green;

		uint8_t views;

		PlyPoint(float x, float y, float z, uint8_t red, uint8_t green, uint8_t blue, uint8_t views) : x(x), y(y), z(z), red(red), green(green), blue(blue), views(views) {}
		
	};

	class PlyExtra {
	public:
		float nx;
		float ny;
		float nz;

		PlyExtra(float nx, float ny, float nz) : nx(nx), ny(ny), nz(nz) {}
	};

	class PlyFile {
	public:
		std::unique_ptr<std::vector<PlyExtra>> extras;
		std::unique_ptr<std::vector<PlyPoint>> points;

		PlyFile(const std::string& path, const std::function<bool(const float x, const float y, const float z)> filter = nullptr) {

			this->extras = std::make_unique<std::vector<PlyExtra>>();
			this->points = std::make_unique<std::vector<PlyPoint>>();

			std::ifstream reader(path, std::ifstream::binary);

			if (!reader.is_open())
				throw std::invalid_argument(std::string("Cannot open file ") + path);

			std::string line;

			std::getline(reader, line);
			if (line != "ply")
				throw std::invalid_argument("Invalid PLY file");

			std::getline(reader, line);
			if (line != "format binary_little_endian 1.0")
				throw std::invalid_argument("Unsupported PLY: only binary little endian files are supported");


			// Skip comments
			do {
				std::getline(reader, line);

				if (line.find("element") == 0)
					break;
				else if (line.find("comment") == 0)
					continue;
				else
					throw std::invalid_argument("Invalid PLY file");

			} while (true);

			// Split line into tokens
			std::vector<std::string> tokens;

			std::istringstream iss(line);
			std::string token;
			while (std::getline(iss, token, ' '))
				tokens.push_back(token);

			if (tokens.size() != 3)
				throw std::invalid_argument("Invalid PLY file");

			if (tokens[0] != "element" && tokens[1] != "vertex")
				throw std::invalid_argument("Invalid PLY file");

			const auto count = std::stoi(tokens[2]);

			std::vector<std::string> properties;

			// Read properties
			do {
				std::getline(reader, line);

				if (line.find("property float x") == 0 ||
					line.find("property float y") == 0 ||
					line.find("property float z") == 0 ||
					line.find("property float nx") == 0 ||
					line.find("property float ny") == 0 ||
					line.find("property float nz") == 0 ||
					line.find("property uchar red") == 0 ||
					line.find("property uchar blue") == 0 ||
					line.find("property uchar green") == 0 ||
					line.find("property uchar views") == 0) {

					properties.push_back(line.substr(15));

				}
				else if (line.find("end_header") == 0)
					break;
				else
					throw std::invalid_argument("Unsupported PLY file: only (x, y, z, [nx, ny, nz], red, green, blue, views) properties");

			} while (true);

			if (properties.size() != 7 && properties.size() != 10)
				throw std::invalid_argument("Unsupported PLY file: only (x, y, z, [nx, ny, nz], red, green, blue, views) properties");

			bool hasNormals = properties.size() == 10;

			points->reserve(count);
			
			if (hasNormals) {

				extras->reserve(count);

				if (filter) {

					// Read points
					for (auto i = 0; i < count; i++) {

						float x, y, z;
						float nx, ny, nz;
						uint8_t red, green, blue;
						uint8_t views;

						reader.read(reinterpret_cast<char*>(&x), sizeof(float));
						reader.read(reinterpret_cast<char*>(&y), sizeof(float));
						reader.read(reinterpret_cast<char*>(&z), sizeof(float));

						reader.read(reinterpret_cast<char*>(&nx), sizeof(float));
						reader.read(reinterpret_cast<char*>(&ny), sizeof(float));
						reader.read(reinterpret_cast<char*>(&nz), sizeof(float));

						reader.read(reinterpret_cast<char*>(&red), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&blue), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&green), sizeof(uint8_t));

						reader.read(reinterpret_cast<char*>(&views), sizeof(uint8_t));

						if (filter(x, y, z)) {
							points->emplace_back(x, y, z, red, green, blue, views);
							extras->emplace_back(nx, ny, nz);
						}
					}
					
				}
				else {

					// Read points
					for (auto i = 0; i < count; i++) {

						float x, y, z;
						float nx, ny, nz;
						uint8_t red, green, blue;
						uint8_t views;

						reader.read(reinterpret_cast<char*>(&x), sizeof(float));
						reader.read(reinterpret_cast<char*>(&y), sizeof(float));
						reader.read(reinterpret_cast<char*>(&z), sizeof(float));

						reader.read(reinterpret_cast<char*>(&nx), sizeof(float));
						reader.read(reinterpret_cast<char*>(&ny), sizeof(float));
						reader.read(reinterpret_cast<char*>(&nz), sizeof(float));

						reader.read(reinterpret_cast<char*>(&red), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&blue), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&green), sizeof(uint8_t));

						reader.read(reinterpret_cast<char*>(&views), sizeof(uint8_t));

						points->emplace_back(x, y, z, red, green, blue, views);
						extras->emplace_back(nx, ny, nz);

					}

				}

			} else {

				if (filter) {

					// Read points
					for (auto i = 0; i < count; i++) {

						float x, y, z;
						uint8_t red, green, blue;
						uint8_t views;

						reader.read(reinterpret_cast<char*>(&x), sizeof(float));
						reader.read(reinterpret_cast<char*>(&y), sizeof(float));
						reader.read(reinterpret_cast<char*>(&z), sizeof(float));

						reader.read(reinterpret_cast<char*>(&red), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&blue), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&green), sizeof(uint8_t));

						reader.read(reinterpret_cast<char*>(&views), sizeof(uint8_t));

						if (filter(x, y, z)) {
							points->emplace_back(x, y, z, red, green, blue, views);
						}
					}
					
				}
				else {

					// Read points
					for (auto i = 0; i < count; i++) {

						float x, y, z;
						uint8_t red, green, blue;
						uint8_t views;

						reader.read(reinterpret_cast<char*>(&x), sizeof(float));
						reader.read(reinterpret_cast<char*>(&y), sizeof(float));
						reader.read(reinterpret_cast<char*>(&z), sizeof(float));

						reader.read(reinterpret_cast<char*>(&red), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&blue), sizeof(uint8_t));
						reader.read(reinterpret_cast<char*>(&green), sizeof(uint8_t));

						reader.read(reinterpret_cast<char*>(&views), sizeof(uint8_t));

						points->emplace_back(x, y, z, red, green, blue, views);

					}					
				}
			}

			reader.close();

		}
	};
}