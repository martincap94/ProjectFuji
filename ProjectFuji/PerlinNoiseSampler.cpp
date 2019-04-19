#include "PerlinNoiseSampler.h"

#include <iostream>
#include <fstream>

using namespace std;

int PerlinNoiseSampler::p[512];

PerlinNoiseSampler::PerlinNoiseSampler() {
}


PerlinNoiseSampler::~PerlinNoiseSampler() {
}

float PerlinNoiseSampler::getSample(glm::vec3 pos, float frequency, bool normalized, bool turbulent) {
	return getSample(pos.x, pos.y, pos.z);
}

float PerlinNoiseSampler::getSample(float x, float y, float z, float frequency, bool normalized, bool turbulent) {
	x *= frequency;
	y *= frequency;
	z *= frequency;

	int X = (int)x & 255;
	int Y = (int)y & 255;
	int Z = (int)z & 255;

	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	float u = fade(x);
	float v = fade(y);
	float w = fade(z);

	int A = p[X] + Y;
	int AA = p[A] + Z;
	int AB = p[A + 1] + Z;
	int B = p[X + 1] + Y;
	int BA = p[B] + Z;
	int BB = p[B + 1] + Z;

	float val = lerp(w, lerp(v, lerp(u,  grad(p[AA], x, y, z),
									grad(p[BA], x - 1, y, z)),
							lerp(u, grad(p[AB], x, y - 1, z),
									grad(p[BB], x - 1, y - 1, z))),
					lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
									grad(p[BA + 1], x - 1, y, z - 1)),
							lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
									grad(p[BB + 1], x - 1, y - 1, z - 1))));
	if (turbulent) {
		return abs(val);
	} else if (normalized) {
		return val * 0.5f + 0.5f;
	}



}

float PerlinNoiseSampler::getSampleOctaves(float x, float y, float z, float startFrequency, bool normalized, int numOctaves, float persistence, bool turbulent) {
	float frequency = startFrequency;
	float amplitude = 1.0f;

	float maxTotalValue = 0.0f;
	float totalValue = 0.0f;
	for (int i = 0; i < numOctaves; i++) {
		totalValue += getSample(x, y, z, frequency, normalized, turbulent) * amplitude;
		maxTotalValue += amplitude;
		amplitude *= persistence;
		frequency *= 2.0f;
	}
	return (totalValue / maxTotalValue);
}


void PerlinNoiseSampler::loadPermutationsData(string filename) {
	ifstream in(filename);
	int val;
	int idx = 0;
	while (in >> val) {
		p[idx] = val;
		p[idx + 256] = val;
		idx++;
	}
	/*
	for (int i = 0; i < 512; i++) {
		cout << p[i] << " ";
		if (i == 255) {
			cout << "| ";
		}
	}
	cout << endl;
	*/
	in.close();
}

float PerlinNoiseSampler::fade(float t) {
	return t * t * t * (t * (t * 6 - 15) + 10);
}

float PerlinNoiseSampler::lerp(float t, float a, float b) {
	return a + t * (b - a);
}

float PerlinNoiseSampler::grad(int hash, float x, float y, float z) {
	int h = hash & 15;
	float u = (h < 8) ? x : y;
	float v = (h < 4) ? y : ((h == 12 || h == 14) ? x : z);
	return (((h & 1) == 0) ? u : -u) + (((h & 2) == 0) ? v : -v);
}
