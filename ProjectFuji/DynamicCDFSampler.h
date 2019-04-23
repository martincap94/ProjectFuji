#pragma once
#include "CDFSampler.h"

#include "PerlinNoiseSampler.h"

class DynamicCDFSampler : public CDFSampler {
public:

	PerlinNoiseSampler pSampler;
	float perlinProbabilityDecrease = 0.4f;

	DynamicCDFSampler(std::string probabilityTexturePath);
	~DynamicCDFSampler();

	void updatePerlinNoiseNaiveTestingCPU(bool onlyPerlin = false);


protected:


	float *arr = nullptr;

	float *d_arr;
	// since we want to change the original array, we cannot do the prefix sum (scan) in-place
	float *d_sums;

	size_t bsize;


	void updateSumsGPU(bool memcpyArrHostToDevice = false);
	void initCUDA();


};

