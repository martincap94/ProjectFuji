///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFSamplerMultiChannel.h
* \author     Martin Cap
*
*	This file describes the CDFSamplerMultiChannel class. The sampler is an extension of regular
*	CDF sampler that uses all texture channels as individual samplers.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>

#include <random>

//! Multi-channel version of the CDFSampler.
/*!
	Multi-channel version of the CDFSampler which is used to sample terrain's materialMap for example.
	Basically, it is 4 CDF samplers that have lower (8-bit) precision probability textures.
*/
class CDFSamplerMultiChannel {
public:

	//! Initialize the mersenne-twister only.
	CDFSamplerMultiChannel();

	//! Constructs the sampler and initializes it.
	/*!
		Constructs the sampler and initializes it with the given probability texture.
		\param[in] probabilityTexturePath	Path to the probability texture (8-bit per channel RGBA assumed).
	*/
	CDFSamplerMultiChannel(std::string probabilityTexturePath);

	//! Destroys the allocated heap arrays.
	~CDFSamplerMultiChannel();

	//! Returns a sample for the specified channel.
	/*!
		Returns a sample for the specified channel.
		If not initialized, returns (0,0). Uses CPU binary search.

		\param[in] channel	The channel of the probability texture to be used for sampling.
		\return Generated sample, (0,0) if not initialized.
	*/
	glm::ivec2 getSample(int channel);

protected:

	float *sums[4] = { nullptr, nullptr, nullptr, nullptr };	//!< Array of pointers to individual sum arrays
	float maxTotalSums[4];	//!< Max total sum for each channel

	int width;					//!< Width of the sampler texture
	int height;					//!< Height of the sampler texture
	int numChannels;			//!< Number of channels of the sampler texture
	int size;					//!< Size of the sampler texture (width * height)

	bool initialized = false;	//!< Whether the sampler is initialized (sums array generated)

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist[4]; //!< Uniform real distributions in range [1, maxTotalSum_channel]

	//!< Initializes the sampler.
	/*!
		Initializes the sampler by loading the texture and immediately computing inclusive prefix sums for each channel individually.
		\param[in] probabilityTexturePath	Path to the probability texture (8-bit per-channel RGBA assumed).
	*/	
	void init(std::string probabilityTexturePath);


};

