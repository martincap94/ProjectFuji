#include "CDFSampler.h"

#include <glad\glad.h>
#include <stb_image.h>

#include <iostream>

#include "TextureManager.h"
#include "Utils.h"
#include "CUDAUtils.cuh"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


using namespace std;

CDFSampler::CDFSampler(std::string probabilityTexturePath) {

	mt = mt19937_64(rd());



	//std::uniform_int_distribution<unsigned long long int> idist;

	stbi_set_flip_vertically_on_load(true);
	//stbi_set_flo

	unsigned short *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}

	sums = new float[width * height]();
	arr = new float[width * height]();
	float *fimgData = new float[width * height]();

	float currSum = 0;
	float maxIntensity = (float)numeric_limits<unsigned short>().max();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += (float)val;
			fimgData[x + y * width] = (float)val / maxIntensity;
			sums[x + y * width] = currSum; // simple sequential inclusive scan (sequential prefix sum)
			arr[x + y * width] = (float)val;
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

	initCUDA();
}

CDFSampler::~CDFSampler() {
	delete[] sums;
}

glm::ivec2 CDFSampler::getSample() {

	int left = 0;
	int right = width * height - 1;

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

void CDFSampler::initCUDA() {

	size_t bsize = sizeof(float) * width * height;
	CHECK_ERROR(cudaMalloc((void**)&d_sums, bsize));
	CHECK_ERROR(cudaMalloc((void**)&d_arr, bsize));

	/*
	// testing
	float maxIntensity = (float)numeric_limits<unsigned short>().max();
	for (int x = 10; x < 100; x++) {
	for (int y = 10; y < 100; y++) {
	arr[x + y * width] = maxIntensity;
	}
	}
	*/


	CHECK_ERROR(cudaMemcpy(d_sums, sums, bsize, cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(d_arr, arr, bsize, cudaMemcpyHostToDevice));


	// now let's test the prefix sum scan from Thrust
	thrust::inclusive_scan(thrust::device, d_arr, d_arr + width * height, d_sums);


	CHECK_ERROR(cudaMemcpy(sums, d_sums, bsize, cudaMemcpyDeviceToHost));





}
