#pragma once
#include "CDFSampler.h"
class DynamicCDFSampler : public CDFSampler {
public:

	DynamicCDFSampler(std::string probabilityTexturePath);
	~DynamicCDFSampler();

	void update(bool memcpyArrHostToDevice = false);

protected:

	float *arr = nullptr;

	float *d_arr;
	// since we want to change the original array, we cannot do the prefix sum (scan) in-place
	float *d_sums;

	size_t bsize;


	void initCUDA();


};

