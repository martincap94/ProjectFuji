///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFSamplerMultiChannel.h
* \author     Martin Cap
* \brief      Describes CDFSamplerMultiChannel class.
*
*	This file describes the CDFSamplerMultiChannel class. The sampler is an extension of regular
*	CDF sampler that uses all texture channels as individual samplers.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>

#include <random>

class CDFSamplerMultiChannel {
public:

	CDFSamplerMultiChannel();

	CDFSamplerMultiChannel(std::string probabilityTexturePath);
	~CDFSamplerMultiChannel();

	glm::ivec2 getSample(int channel);

protected:

	float *sums[4] = { nullptr, nullptr, nullptr, nullptr };
	float maxTotalSums[4];

	int width;
	int height;
	int numChannels;
	int size;

	bool initialized = false;

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist[4];

	void init(std::string probabilityTexturePath);


};

