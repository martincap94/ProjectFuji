#pragma once

#include <glm\glm.hpp>

#include <random>

class CDFSampler {
public:



	float *arr = nullptr;
	float *sums = nullptr;
	float maxTotalSum = 0;

	int width;
	int height;
	int numChannels;



	// expects path to 16-bit grayscale png
	CDFSampler(std::string probabilityTexturePath);
	~CDFSampler();

	glm::ivec2 getSample();

private:

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist;


	float *d_arr;
	// since we want to change the original array, we cannot do the prefix sum (scan) in-place
	float *d_sums;

	void initCUDA();

};

