#include "CDFEmitterCUDA.h"

#include "ParticleSystem.h"
#include "TextureManager.h"
#include "Utils.h"

#include <stb_image.h>

using namespace std;

// expects path to 16-bit grayscale png
CDFEmitterCUDA::CDFEmitterCUDA(ParticleSystem *owner, string probabilityTexturePath) : Emitter(owner) {

	//std::uniform_int_distribution<unsigned long long int> idist;

	stbi_set_flip_vertically_on_load(false);


	unsigned short *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}

	sums = new float[width * height]();
	float *fimgData = new float[width * height]();

	float currSum = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += (float)val;
			sums[x + y * width] = currSum; // simple sequential inclusive scan (sequential prefix sum)
		}
	}
	maxTotalSum = currSum;

	cout << "Max total sum = " << maxTotalSum << endl;

	firstdist = uniform_real_distribution<float>(1, maxTotalSum);

	for (int i = 0; i < width * height; i++) {
		fimgData[i] = sums[i] / maxTotalSum;
	}




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


}


CDFEmitterCUDA::~CDFEmitterCUDA() {
	delete[] sums;
}

void CDFEmitterCUDA::emitParticle() {

	preEmitCheck();

	int selectedRow = height - 1;
	int selectedCol = width - 1;


	int left = 0;
	int right = width * height - 1;

	float randVal = firstdist(mt);

	int idx;
	while (left <= right) {
		idx = (left + right) / 2;
		if (randVal <= sums[idx]) {
			right = idx - 1;
		} else {
			left = idx + 1;
		}
	}
	idx = left;

	selectedRow = idx / width;
	selectedCol = idx % width;


	Particle p;
	glm::vec3 pos;

	p.profileIndex = 1;
	p.velocity = glm::vec3(0.0f);

	//cout << pos.x << ", " << pos.y << ", " << pos.z << endl;

	//for (int i = 0; i < 1000; i++) {
		pos = glm::vec3(selectedRow, 0.0f, selectedCol);

		// move inside the texel
		pos.x += getRandFloat(0.0f, 1.0f);
		pos.z += getRandFloat(0.0f, 1.0f);

		pos.x *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
		pos.z *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
		pos.y = owner->heightMap->getHeight(pos.x, pos.z, true);
		p.position = pos;


		owner->pushParticleToEmit(p);
	//}

}

void CDFEmitterCUDA::update() {
}

void CDFEmitterCUDA::draw() {
}

void CDFEmitterCUDA::draw(ShaderProgram * shader) {
}

void CDFEmitterCUDA::initBuffers() {
}
