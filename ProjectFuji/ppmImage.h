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

//! Helper class for loading ASCII .ppm images.
/*!
	Use .ppm images very sparingly. They are mainly good for initial tests.
*/
class ppmImage {
public:

	int width;			//!< Width of the image
	int height;			//!< Height of the image
	int maxIntensity;	//!< Maximum texel intensity (should be 255 for .ppm images)

	glm::vec3 **data;	//!< Texture data stored as 2D array for ease of use

	//! Loads the .ppm image from the given filename.
	/*!
		ASCII format expected!
		\param[in] filename		Filename of the image.
	*/
	ppmImage(std::string filename);

	//! Frees the texture data.
	~ppmImage();



};

