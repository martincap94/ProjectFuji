///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFSampler.h
* \author     Martin Cap
* \brief      Describes emitter that uses CDF sampler.
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

class CDFSampler {
public:

	CDFSampler();
	// expects path to 16-bit grayscale png
	CDFSampler(std::string probabilityTexturePath);
	~CDFSampler();

	glm::ivec2 getSample();

	int getWidth();
	int getHeight();

	Texture *getTexture();

protected:

	float *sums = nullptr;
	float maxTotalSum = 0;

	int width;
	int height;
	int numChannels;
	int size;

	Texture *tex = nullptr; // texture loaded in TextureManager - stored in OpenGL for drawing

	bool initialized = false;

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist;

	void init(std::string probabilityTexturePath);


};

