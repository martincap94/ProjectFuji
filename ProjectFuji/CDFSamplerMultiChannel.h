#pragma once

#include <glm\glm.hpp>

#include <random>

class CDFSamplerMultiChannel {
public:

	CDFSamplerMultiChannel();
	// expects path to 16-bit grayscale png
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

