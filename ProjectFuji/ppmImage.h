#pragma once

#include <string>
#include <glm\glm.hpp>

class ppmImage {
public:

	int width;
	int height;
	int maxIntensity;

	glm::vec3 **data;

	ppmImage(std::string filename);
	~ppmImage();



};

