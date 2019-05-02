///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       ppmImage.h
* \author     Martin Cap
*
*	Helper class that loads images in .ppm format. Easy format for testing, do not use in production.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
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

