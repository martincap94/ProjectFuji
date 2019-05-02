///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFSampler.h
* \author     Martin Cap
*
*	This file describes the CDFEmitter class that uses CDFSampler class to generate particles in a
*	pattern given by a probability grayscale texture.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Texture.h"
#include <glm\glm.hpp>

#include <random>

//! 2D sampler that uses CDF with probability textures.
/*!
	2D sampler that uses CDF and a probability texture to generate particles in a given pattern.
	The probability texture is assumed to be grayscale (16-bit per channel) texture.
	Beware that resolution of the texture is important if we want to use this in world coordinate system.
*/
class CDFSampler {
public:

	//! Default constructor that initializes the mersenne twister engine.
	CDFSampler();
	//! Constructs the sampler with the given probability texture.
	/*!
		Constructs the sampler with the given probability texture - expects path to 16-bit grayscale (png).
		Initializes the sampler.
	*/
	CDFSampler(std::string probabilityTexturePath);

	//! Destroys the allocated array memory.
	~CDFSampler();

	//! Returns one generated sample. Returns (0,0) if not initialized.
	/*!
		Returns one generated sample. Returns (0,0) if not initialized.
		Uses CPU binary search (if more values equal, finds the leftmost one).
		GPU binary search also possible (use define in source file) - much much slower.
		\return Generated random sample from the texture distribution.
	*/
	glm::ivec2 getSample();

	//! Returns width of the sampler texture.
	int getWidth();
	//! Returns height of the sampler texture.
	int getHeight();

	//! Returns the sampler texture.
	Texture *getTexture();

protected:

	float *sums = nullptr;		//!< Flattened array of inclusive prefix sums
	float maxTotalSum = 0;		//!< Maximum total sum (last member of the inclusive(!) prefix sum array)

	int width;					//!< Width of the sampler texture
	int height;					//!< Height of the sampler texture
	int numChannels;			//!< Number of channels of the sampler texture
	int size;					//!< Size of the sampler texture (width * height)

	Texture *tex = nullptr;		//!< Helper texture loaded in TextureManager - stored in OpenGL for drawing

	bool initialized = false;	//!< Whether the sampler is initialized (sums array generated)

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist;	//!< Uniform real distribution in range [1, maxTotalSum]

	//!< Initializes the sampler.
	/*!
		Initializes the sampler by loading the texture and immediately computing inclusive prefix sums.
		\param[in] probabilityTexturePath	Path to the probability texture.
	*/
	void init(std::string probabilityTexturePath);


};

