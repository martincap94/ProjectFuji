#include "CDFSamplerMultiChannel.h"

#include <glad\glad.h>
#include <stb_image.h>

#include <iostream>

#include "TextureManager.h"
#include "Utils.h"


using namespace std;

CDFSamplerMultiChannel::CDFSamplerMultiChannel() {
	mt = mt19937_64(rd());
}

CDFSamplerMultiChannel::CDFSamplerMultiChannel(string probabilityTexturePath) : CDFSamplerMultiChannel() {
	init(probabilityTexturePath);
}

CDFSamplerMultiChannel::~CDFSamplerMultiChannel() {
	for (int i = 0; i < numChannels; i++) {
		if (sums[i]) {
			delete[] sums[i];
		}
	}
	
}

glm::ivec2 CDFSamplerMultiChannel::getSample(int channel) {
	if (!initialized) {
		return glm::ivec2(0);
	}

	int left = 0;
	int right = size - 1;

	float randVal = firstdist[channel](mt);

	int idx;

	while (left <= right) {
		idx = (left + right) / 2;
		if (randVal <= sums[channel][idx]) {
			right = idx - 1;
		} else {
			left = idx + 1;
		}
	}
	idx = left;
	//cout << "idx = " << idx << endl;


	int selectedRow = idx / width;
	int selectedCol = idx % width;

	return glm::ivec2(selectedRow, selectedCol);
}




void CDFSamplerMultiChannel::init(string probabilityTexturePath) {

	stbi_set_flip_vertically_on_load(true);

	typedef unsigned short imgtype;
	imgtype *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}

	size = width * height;

	for (int i = 0; i < numChannels; i++) {
		sums[i] = new float[size]();
	}

	float currSums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	float maxIntensity = (float)numeric_limits<imgtype>().max();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			imgtype *pixel = imageData + (x + y * width) * numChannels;
			for (int i = 0; i < numChannels; i++) {
				imgtype val = pixel[i];
				currSums[i] += (float)val;
				sums[i][x + y * width] = currSums[i];
			}
		}
	}
	for (int i = 0; i < numChannels; i++) {
		maxTotalSums[i] = currSums[i];
		firstdist[i] = uniform_real_distribution<float>(1, maxTotalSums[i]);
	}

	if (imageData) {
		stbi_image_free(imageData);
	}

	initialized = true;
}



