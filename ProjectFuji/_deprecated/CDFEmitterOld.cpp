#include "CDFEmitterOld.h"

#include "ParticleSystem.h"
#include "TextureManager.h"
#include "Utils.h"

#include <stb_image.h>

using namespace std;

//#define CDF_2D

// expects path to 16-bit grayscale png
CDFEmitterOld::CDFEmitterOld(ParticleSystem *owner, string probabilityTexturePath) : Emitter(owner) {

	//std::uniform_int_distribution<unsigned long long int> idist;

	stbi_set_flip_vertically_on_load(false);


	unsigned short *imageData = stbi_load_16(probabilityTexturePath.c_str(), &width, &height, &numChannels, NULL);
	if (!imageData) {
		cout << "Error loading texture at " << probabilityTexturePath << endl;
		stbi_image_free(imageData);
		return;
	}


#ifdef CDF_2D
	sums = new unsigned long long int[(width + 1) * height]();
	float *fimgData = new float[width * height]();
	ewidth = width + 1;

	unsigned long long int maxSumFound = 0; // used for normalizing the data for texture visualization (GL_FLOAT)
	unsigned long long int currSum = 0;

	for (int y = 0; y < height; y++) {
		currSum = 0;
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += val;
			sums[x + y * ewidth] = currSum; // simple sequential inclusive scan (sequential prefix sum)
			fimgData[x + y * width] = (float)currSum;
			if (currSum > maxSumFound) {
				maxSumFound = currSum;
			}
		}
	}

	currSum = 0;
	for (int y = 0; y < height; y++) {
		currSum += sums[width - 1 + y * ewidth];
		sums[width + y * ewidth] = currSum;
	}
	maxTotalSum = currSum;

	cout << "Max total sum = " << maxTotalSum << endl;

	firstdist = uniform_int_distribution<unsigned long long int>(1, maxTotalSum);

	float fmaxSumFound = (float)maxSumFound;
	for (int i = 0; i < width * height; i++) {
		fimgData[i] /= fmaxSumFound;
	}
#else
	sums = new unsigned long long int[width * height]();
	float *fimgData = new float[width * height]();

	unsigned long long int currSum = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned short *pixel = imageData + (x + y * width) * numChannels;
			unsigned short val = pixel[0];
			currSum += val;
			sums[x + y * width] = currSum; // simple sequential inclusive scan (sequential prefix sum)
		}
	}
	maxTotalSum = currSum;

	cout << "Max total sum = " << maxTotalSum << endl;

	firstdist = uniform_int_distribution<unsigned long long int>(1, maxTotalSum);

	for (int i = 0; i < width * height; i++) {
		fimgData[i] = ((float)sums[i] / (float)maxTotalSum);
		//cout << fimgData[i] << ", ";
 	}



#endif


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


CDFEmitterOld::~CDFEmitterOld() {
	delete[] sums;
}

void CDFEmitterOld::emitParticle() {

	if (!canEmitParticle()) {
		return;
	}


	int selectedRow = height - 1;
	int selectedCol = width - 1;



#ifdef CDF_2D

	unsigned long long int randVal = firstdist(mt);

	//cout << "randVal for COL = " << randVal << endl;

	// find row
	for (int y = 0; y < height; y++) {
		if (randVal <= sums[width + y * ewidth]) {
			selectedRow = y; // we cannot get zero since it cannot hold that 0 < 0
			break;
		}
	}
	//cout << "selected row = " << selectedRow << endl;

	unsigned long long int maxRowVal = sums[width - 1 + selectedRow * ewidth];
	//cout << "maxRowVal = " << maxRowVal << endl;
	secondDist = uniform_int_distribution<unsigned long long int>(1, maxRowVal);
	randVal = secondDist(mt);

	//cout << "randVal for ROW = " <<  randVal << endl;

	for (int x = 0; x < width; x++) {
		//cout << " | comp: " << randVal << " < " << sums[x + selectedRow * ewidth] << endl;
		if (randVal <= sums[x + selectedRow * ewidth]) {
			selectedCol = x;
			break;
		}
	}

	//cout << "row = " << selectedRow << ", col = " << selectedCol << endl;
#else


	int left = 0;
	int right = width * height - 1;

	unsigned long long int randVal = firstdist(mt);

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


#endif

	Particle p;
	glm::vec3 pos(selectedRow, 0.0f, selectedCol);
	pos.y = owner->heightMap->getHeight(selectedRow, selectedCol, false);
	pos.x *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
	pos.z *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
	p.position = pos;
	p.profileIndex = 1;
	p.velocity = glm::vec3(0.0f);

	//cout << pos.x << ", " << pos.y << ", " << pos.z << endl;

	for (int i = 0; i < 10; i++) {
		owner->pushParticleToEmit(p);
	}

}

void CDFEmitterOld::update() {
}

void CDFEmitterOld::draw() {
}

void CDFEmitterOld::draw(ShaderProgram * shader) {
}

void CDFEmitterOld::initBuffers() {
}
