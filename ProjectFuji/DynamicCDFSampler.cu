#include "DynamicCDFSampler.h"

#include "CUDAUtils.cuh"

#include <glad\glad.h>
#include <stb_image.h>

#include <iostream>

#include "TextureManager.h"
#include "Utils.h"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

using namespace std;

DynamicCDFSampler::DynamicCDFSampler(string probabilityTexturePath) : CDFSampler() {


	stbi_set_flip_vertically_on_load(true);

	unsigned short *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}
	size = width * height;
	bsize = sizeof(float) * size;


	sums = new float[size]();
	arr = new float[size]();

	float currSum = 0;
	//float maxIntensity = (float)numeric_limits<unsigned short>().max();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += (float)val;
			sums[x + y * width] = currSum; // simple sequential inclusive scan (sequential prefix sum)
			arr[x + y * width] = (float)val;
		}
	}
	maxTotalSum = currSum;

	//cout << "Max total sum = " << maxTotalSum << endl;

	firstdist = uniform_real_distribution<float>(1, maxTotalSum);


	if (imageData) {
		stbi_image_free(imageData);
	}

	initCUDA();

	initialized = 1;


}


DynamicCDFSampler::~DynamicCDFSampler() {
	CDFSampler::~CDFSampler();
	if (arr) {
		delete[] arr;
	}
	CHECK_ERROR(cudaFree(d_arr));
	CHECK_ERROR(cudaFree(d_sums));
}

void DynamicCDFSampler::updatePerlinNoiseCPU(bool onlyPerlin) {
	float currSum = 0.0f;
	float currSeed = 0.0f;
	if (useTimeAsSeed) {
		currSeed = (float)glfwGetTime();
	} else {
		currSeed = (float)this->seed;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float val = 0.0f;
			if (onlyPerlin) {
				val = glm::clamp(pSampler.getSampleOctaves((float)x / (float)width, (float)y / (float)height, currSeed) - perlinProbabilityDecrease, 0.0f, 1.0f);
			} else {
				val = arr[x + y * width] * glm::clamp(pSampler.getSampleOctaves((float)x / (float)width, (float)y / (float)height, currSeed) - perlinProbabilityDecrease, 0.0f, 1.0f);
			}
			currSum += val;
			sums[x + y * width] = currSum;
			//arr[x + y * width] = (float)val;
		}
	}
	maxTotalSum = currSum;
	firstdist = uniform_real_distribution<float>(1, maxTotalSum);


}

void DynamicCDFSampler::updateSumsGPU(bool memcpyArrHostToDevice) {


	if (memcpyArrHostToDevice) {
		CHECK_ERROR(cudaMemcpy(d_arr, arr, bsize, cudaMemcpyHostToDevice));
	}

	thrust::inclusive_scan(thrust::device, d_arr, d_arr + width * height, d_sums);

	CHECK_ERROR(cudaMemcpy(sums, d_sums, bsize, cudaMemcpyDeviceToHost));
}

void DynamicCDFSampler::initCUDA() {

	CHECK_ERROR(cudaMalloc((void**)&d_sums, bsize));
	CHECK_ERROR(cudaMalloc((void**)&d_arr, bsize));


}
