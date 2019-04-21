#pragma once

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

protected:

	float *sums = nullptr;
	float maxTotalSum = 0;

	int width;
	int height;
	int numChannels;
	int size;

	bool initialized = false;

	std::random_device rd;
	std::mt19937_64 mt;

	std::uniform_real_distribution<float> firstdist;

	void init(std::string probabilityTexturePath);


};

