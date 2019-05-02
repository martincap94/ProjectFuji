///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PerlinNoiseSampler.h
* \author     Martin Cap
*
*	This file describes the 3D perlin noise sampler that provides both static and instance options.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glm\glm.hpp>
#include <string>

#include <nuklear.h>

class PerlinNoiseSampler {
public:

	enum eSamplingMode {
		BASIC = 0,
		NORMALIZED,
		TURBULENT,
		_NUM_MODES

	};

	float frequency = 1.0f;
	int numOctaves = 1;
	float persistence = 0.5f;
	int samplingMode = NORMALIZED;


	PerlinNoiseSampler();
	~PerlinNoiseSampler();

	float getSample(glm::vec3 pos);
	float getSample(float x, float y, float z);
	float getSampleOctaves(glm::vec3 pos);
	float getSampleOctaves(float x, float y, float z);


	// beware that when x and y are integers, this noise function always returns 0.0!
	static float getSampleStatic(glm::vec3 pos, float frequency = 1.0f, int samplingMode = 1);
	// beware that when x and y are integers, this noise function always returns 0.0!
	static float getSampleStatic(float x, float y, float z, float frequency = 1.0f, int samplingMode = 1);
	static float getSampleOctavesStatic(glm::vec3 pos, float startFrequency = 1.0f, int numOctaves = 1, float persistence = 0.5f, int samplingMode = 1);
	static float getSampleOctavesStatic(float x, float y, float z, float startFrequency = 1.0f, int numOctaves = 1, float persistence = 0.5f, int samplingMode = 1);

	static void loadPermutationsData(std::string filename);


	void constructUIPropertiesTab(struct nk_context *ctx);

	const char *getSamplingModeString();
	const char *getSamplingModeString(int samplingMode);

private:

	static int p[512];

	static float fade(float t);
	static float lerp(float t, float a, float b);
	static float grad(int hash, float x, float y, float z);


};

