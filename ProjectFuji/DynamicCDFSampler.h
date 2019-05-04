///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       DynamicCDFSampler.h
* \author     Martin Cap
*
*	Description of the DynamicCDFSampler class. It is a subclass of the CDFSampler. It offers
*	recalculation of the prefix sum scan on GPU.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "CDFSampler.h"

#include "PerlinNoiseSampler.h"

//! CDFSampler that can modify the probability texture at runtime computing prefix sum on GPU.
/*!
	Uses Thrust library to compute parallel prefix sum on the GPU.
	The final sampling is done on the CPU because the binary search is much faster (it is not an
	easily parallelizable problem, especially for arrays of variable length).
*/
class DynamicCDFSampler : public CDFSampler {
public:

	PerlinNoiseSampler pSampler;				//!< Perlin noise sampler that can be used as the probability texture generator
	float perlinProbabilityDecrease = 0.4f;		//!< Artificial value that is subtracted from the generated perlin samples to generate more pronounced areas

	int useTimeAsSeed = 1;	//!< Whether to use time as seed for the perlin sampler
	int seed = 0;			//!< Seed for the perlin sampler


	//! Loads the probability texture and precomputes initial prefix sums.
	/*!
		\param[in] probabilityTexturePath		Path to the probability texture file.
	*/
	DynamicCDFSampler(std::string probabilityTexturePath);

	//! Frees the CPU and GPU allocated data.
	~DynamicCDFSampler();

	//! Updates the sums using the perlin noise instance on CPU.
	/*!
		\param[in] onlyPerlin	Whether we want to generate new probability texture using only perlin noise or we just want
		to multiply the original probability texture with the generated perlin noise.
	*/
	void updatePerlinNoiseCPU(bool onlyPerlin = false);


protected:


	float *arr = nullptr;	//!< CPU 1D array of the probability texture

	float *d_arr;			//!< GPU 1D array of the probability texture
	// since we want to change the original array, we cannot do the prefix sum (scan) in-place
	float *d_sums;			//!< GPU 1D array of the prefix sums

	size_t bsize;			//!< Size of the probability texture array in bytes

	//! Updates the sum array using the GPU.
	/*!
		\param[in] memcpyArrHostToDevice	Whether to copy probability texture to array to GPU first.
	*/
	void updateSumsGPU(bool memcpyArrHostToDevice = false);

	//! Allocates CUDA memory for the d_arr and d_sums 1D arrays.
	void initCUDA();


};

