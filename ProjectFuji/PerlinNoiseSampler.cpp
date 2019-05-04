#include "PerlinNoiseSampler.h"

#include <iostream>
#include <fstream>

using namespace std;

int PerlinNoiseSampler::p[512];

PerlinNoiseSampler::PerlinNoiseSampler() {}


PerlinNoiseSampler::~PerlinNoiseSampler() {}

float PerlinNoiseSampler::getSample(glm::vec3 pos) {
	return getSample(pos.x, pos.y, pos.z);
}

float PerlinNoiseSampler::getSample(float x, float y, float z) {
	return getSampleStatic(x, y, z, frequency, samplingMode);
}

float PerlinNoiseSampler::getSampleOctaves(glm::vec3 pos) {
	return getSampleOctaves(pos.x, pos.y, pos.z);
}

float PerlinNoiseSampler::getSampleOctaves(float x, float y, float z) {
	return getSampleOctavesStatic(x, y, z, frequency, numOctaves, persistence, samplingMode);
}



float PerlinNoiseSampler::getSampleStatic(glm::vec3 pos, float frequency, int samplingMode) {
	return getSampleStatic(pos.x, pos.y, pos.z);
}

float PerlinNoiseSampler::getSampleStatic(float x, float y, float z, float frequency, int samplingMode) {
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

	float val = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
										grad(p[BA], x - 1, y, z)),
								lerp(u, grad(p[AB], x, y - 1, z),
										grad(p[BB], x - 1, y - 1, z))),
						lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
										grad(p[BA + 1], x - 1, y, z - 1)),
								lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
										grad(p[BB + 1], x - 1, y - 1, z - 1))));

	switch (samplingMode) {
		case NORMALIZED:
			return val * 0.5f + 0.5f;
		case TURBULENT:
			return abs(val);
		case BASIC:
		default:
			return val;
	}
}

float PerlinNoiseSampler::getSampleOctavesStatic(glm::vec3 pos, float startFrequency, int numOctaves, float persistence, int samplingMode) {
	return getSampleOctavesStatic(pos.x, pos.y, pos.z, startFrequency, numOctaves, persistence, samplingMode);
}

float PerlinNoiseSampler::getSampleOctavesStatic(float x, float y, float z, float startFrequency, int numOctaves, float persistence, int samplingMode) {

	if (numOctaves <= 1) {
		return getSampleStatic(x, y, z, startFrequency, samplingMode);
	}

	float frequency = startFrequency;
	float amplitude = 1.0f;

	float maxTotalValue = 0.0f;
	float totalValue = 0.0f;
	for (int i = 0; i < numOctaves; i++) {
		totalValue += getSampleStatic(x, y, z, frequency, samplingMode) * amplitude;
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

void PerlinNoiseSampler::constructUIPropertiesTab(nk_context *ctx) {
	nk_layout_row_dynamic(ctx, 15, 1);
	nk_property_float(ctx, "Frequency", 0.01f, &frequency, 100.0f, 0.01f, 0.01f);
	nk_property_int(ctx, "Num. Octaves", 1, &numOctaves, 10, 1, 1);
	nk_property_float(ctx, "Persistence", 0.01f, &persistence, 1.0f, 0.01f, 0.01f);

	if (nk_combo_begin_label(ctx, getSamplingModeString(), nk_vec2(nk_widget_width(ctx), 100.0f))) {
		for (int i = 0; i < eSamplingMode::_NUM_MODES; i++) {
			nk_layout_row_dynamic(ctx, 15, 1);
			if (nk_combo_item_label(ctx, getSamplingModeString(i), NK_TEXT_CENTERED)) {
				samplingMode = i;
				nk_combo_close(ctx);
			}
		}
		nk_combo_end(ctx);
	}
}

const char * PerlinNoiseSampler::getSamplingModeString() {
	return getSamplingModeString(this->samplingMode);
}

const char * PerlinNoiseSampler::getSamplingModeString(int samplingMode) {
	switch (samplingMode) {
		case eSamplingMode::BASIC:
			return "Basic";
		case eSamplingMode::NORMALIZED:
			return "Normalized to [0, 1]";
		case eSamplingMode::TURBULENT:
			return "Turbulent (abs)";
	}
	return "None";
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
