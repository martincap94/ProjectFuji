#pragma once

#include <glm\glm.hpp>
#include <string>

class PerlinNoiseSampler {
public:

	PerlinNoiseSampler();
	~PerlinNoiseSampler();


	// beware that when x and y are integers, this noise function always returns 0.0!
	static float getSample(glm::vec3 pos, float frequency = 1.0f, bool normalized = true, bool turbulent = false);
	// beware that when x and y are integers, this noise function always returns 0.0!
	static float getSample(float x, float y, float z, float frequency = 1.0f, bool normalized = true, bool turbulent = false);
	 
	static float getSampleOctaves(float x, float y, float z, float startFrequency = 1.0f, bool normalized = true, int numOctaves = 1, float persistence = 0.5f, bool turbulent = false);

	static void loadPermutationsData(std::string filename);

	// add turbulence, repeat, 

private:

	static int p[512];

	static float fade(float t);
	static float lerp(float t, float a, float b);
	static float grad(int hash, float x, float y, float z);


};

