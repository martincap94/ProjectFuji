#include "CDFSampler.h"

#include <glad\glad.h>
#include <stb_image.h>

#include <iostream>

#include "TextureManager.h"
#include "Utils.h"


using namespace std;

CDFSampler::CDFSampler() {
	mt = mt19937_64(rd());
}

CDFSampler::CDFSampler(string probabilityTexturePath) : CDFSampler() {
	init(probabilityTexturePath);
}

CDFSampler::~CDFSampler() {
	if (sums) {
		delete[] sums;
	}
}

glm::ivec2 CDFSampler::getSample() {
	if (!initialized) {
		return glm::ivec2(0);
	}

	int left = 0;
	int right = size - 1;

	float randVal = firstdist(mt);

	int idx;

#ifdef THRUST_BIN_SEARCH // much slower than CPU version
	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_sums);
	idx = thrust::distance(d_sums, thrust::lower_bound(thrust::device, d_sums, d_sums + width * height, randVal));
#else

	while (left <= right) {
		idx = (left + right) / 2;
		if (randVal <= sums[idx]) {
			right = idx - 1;
		} else {
			left = idx + 1;
		}
	}
	idx = left;
#endif
	//cout << "idx = " << idx << endl;


	int selectedRow = idx / width;
	int selectedCol = idx % width;

	return glm::ivec2(selectedRow, selectedCol);
}

int CDFSampler::getWidth() {
	return width;
}

int CDFSampler::getHeight() {
	return height;
}




void CDFSampler::init(string probabilityTexturePath) {

	stbi_set_flip_vertically_on_load(true);

	unsigned short *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}

	size = width * height;
	sums = new float[size]();
	float *fimgData = new float[size]();

	float currSum = 0;
	float maxIntensity = (float)numeric_limits<unsigned short>().max();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += (float)val;
			fimgData[x + y * width] = (float)val / maxIntensity;
			sums[x + y * width] = currSum; // simple sequential inclusive scan (sequential prefix sum)
		}
	}
	maxTotalSum = currSum;

	//cout << "Max total sum = " << maxTotalSum << endl;

	firstdist = uniform_real_distribution<float>(1, maxTotalSum);


	GLuint texId;

	glGenTextures(1, &texId);
	glBindTexture(GL_TEXTURE_2D, texId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, fimgData);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	TextureManager::pushCustomTexture(texId, width, height, 1, "CDF Emitter test");

	delete[] fimgData;

	CHECK_GL_ERRORS();



	if (imageData) {
		stbi_image_free(imageData);
	}

	initialized = true;
}



