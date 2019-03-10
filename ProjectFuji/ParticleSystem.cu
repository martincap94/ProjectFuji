#include "ParticleSystem.h"

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "Utils.h"
#include "LBM.h"
#include "CUDAUtils.cuh"

#include "Emitter.h"
#include "CircleEmitter.h"


ParticleSystem::ParticleSystem(VariableManager *vars) : vars(vars) {

	heightMap = vars->heightMap;
	numParticles = vars->numParticles;
	numActiveParticles = 0;
	//numActiveParticles = numParticles;

	initBuffers();
	initCUDA();

	spriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture.png").c_str());
	secondarySpriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture2.png").c_str());

	emitters.push_back(new CircleEmitter(this, this->heightMap, glm::vec3(10.0f, 0.0f, 10.0f), 2.0f, true));
	emitters.push_back(new CircleEmitter(this, this->heightMap, glm::vec3(0.0f), 5.0f, true));


}


ParticleSystem::~ParticleSystem() {
	//delete[] particleVertices;

	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaParticleVerticesVBO));
	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaParticleProfilesVBO));

	for (int i = 0; i < emitters.size(); i++) {
		delete emitters[i];
	}

	cudaFree(d_numParticles);

}



void ParticleSystem::initBuffers() {

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);
	glGenBuffers(1, &particleVerticesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleVerticesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glGenBuffers(1, &particleProfilesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleProfilesVBO);

	glEnableVertexAttribArray(5);
	glVertexAttribIPointer(5, 1, GL_INT, sizeof(int), (void *)0);

	glBindVertexArray(0);

}


void ParticleSystem::initCUDA() {

	cudaMalloc((void**)&d_numParticles, sizeof(int));
	cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_verticalVelocities, sizeof(float) * numParticles);
	cudaMalloc((void**)&d_profileIndices, sizeof(int) * numParticles);
	//cudaMalloc((void**)&d_particlePressures, sizeof(float) * numParticles);

	cudaMemset(d_verticalVelocities, 0, sizeof(float) * numParticles);
	//cudaMemset(d_profileIndices, 0, sizeof(int) * numParticles);
	//cudaMemset(d_particlePressures, 0, sizeof(float) * numParticles);

}

void ParticleSystem::emitParticles() {

	//// check if emitting particles is possible (maximum reached)
	//if (numActiveParticles >= numParticles) {
	//	return;
	//}
	int prevNumActiveParticles = numActiveParticles;

	// go through all emitters and emit particles (each pushes them to this system)
	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->emitParticle();
	}



	// upload the data to VBOs and CUDA memory

	glNamedBufferSubData(particleVerticesVBO, sizeof(glm::vec3) * prevNumActiveParticles, sizeof(glm::vec3) * particleVerticesToEmit.size()/*(numActiveParticles - prevNumActiveParticles)*/, particleVerticesToEmit.data());

	glNamedBufferSubData(particleProfilesVBO, sizeof(int) * prevNumActiveParticles, sizeof(int) * particleProfilesToEmit.size(), particleProfilesToEmit.data());

	cudaMemcpy(d_verticalVelocities + prevNumActiveParticles, verticalVelocitiesToEmit.data(), (numActiveParticles - prevNumActiveParticles) * sizeof(float), cudaMemcpyHostToDevice);


	// clear the temporary emitted particle structures

	particleVerticesToEmit.clear();
	particleProfilesToEmit.clear();
	verticalVelocitiesToEmit.clear();

}


void ParticleSystem::draw(const ShaderProgram &shader, glm::vec3 cameraPos) {


	glUseProgram(shader.id);

	shader.setInt("u_Tex", 0);
	shader.setInt("u_SecondTex", 1);
	shader.setVec3("u_TintColor", vars->tintColor);

	shader.setInt("u_OpacityBlendMode", opacityBlendMode);
	shader.setFloat("u_OpacityBlendRange", opacityBlendRange);


	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture.id);

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, secondarySpriteTexture.id);

	glPointSize(pointSize);
	shader.setVec3("u_CameraPos", cameraPos);
	shader.setFloat("u_PointSizeModifier", pointSize);
	shader.setFloat("u_OpacityMultiplier", vars->opacityMultiplier);

	glBindVertexArray(particlesVAO);

	glDrawArrays(GL_POINTS, 0, numParticles);

	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->draw();
	}


}

void ParticleSystem::initParticlesWithZeros() {
	cout << __FUNCTION__ << " not yet implemented!" << endl;

	/*
	vector<glm::vec3> particleVertices;

	for (int i = 0; i < numParticles; i++) {
		particleVertices.push_back(glm::vec3(0.0f));
	}

	glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numParticles, particleVertices.data(), GL_STATIC_DRAW);

	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particleVerticesVBO, cudaGraphicsRegisterFlagsWriteDiscard));


	cudaMemset(d_profileIndices, 0, sizeof(int) * numParticles);
	cudaMemset(d_particlePressures, 0, sizeof(float) * numParticles);
	*/
}

void ParticleSystem::initParticlesOnTerrain() {

	vector<glm::vec3> particleVertices;
	vector<int> particleProfiles;
	vector<float> particlePressures;

	ppmImage *profileMap = stlpSim->profileMap;
	STLPDiagram *stlpDiagram = stlpSim->stlpDiagram;

	for (int i = 0; i < numParticles; i++) {
		Particle p;

		// testing generation in circle
		float randx;
		float randz;

		int leftx;
		int rightx;
		int leftz;
		int rightz;

		float xRatio;
		float zRatio;

		if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {

			float recalculationVal = 0.0f;
			glm::vec3 pif(0.0f);
			int numPositionRecalculations = 0;
			do {
				randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
				randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));

				// interpolate
				leftx = (int)randx;
				rightx = leftx + 1;
				leftz = (int)randz;
				rightz = leftz + 1;


				// leftx and leftz cannot be < 0 and rightx and rightz cannot be >= GRID_WIDTH or GRID_DEPTH
				xRatio = randx - leftx;
				zRatio = randz - leftz;

				glm::vec3 p1 = profileMap->data[leftx][leftz];
				glm::vec3 p2 = profileMap->data[leftx][rightz];
				glm::vec3 p3 = profileMap->data[rightx][leftz];
				glm::vec3 p4 = profileMap->data[rightx][rightz];

				glm::vec3 pi1 = zRatio * p2 + (1.0f - zRatio) * p1;
				glm::vec3 pi2 = zRatio * p4 + (1.0f - zRatio) * p3;

				pif = xRatio * pi2 + (1.0f - xRatio) * pi1;
				recalculationVal = pif.z / (float)profileMap->maxIntensity;

				numPositionRecalculations++;

			} while (recalculationVal < positionRecalculationThreshold && numPositionRecalculations < maxPositionRecalculations);

			glm::ivec3 pii = (glm::ivec3)pif;

			if (pii.y != pii.x) {
				p.profileIndex = (rand() % (pii.y - pii.x) + pii.x) % (stlpDiagram->numProfiles - 1);
			} else {
				p.profileIndex = pii.x % (stlpDiagram->numProfiles - 1);
			}

		} else {

			randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
			randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));

			// interpolate
			leftx = (int)randx;
			rightx = leftx + 1;
			leftz = (int)randz;
			rightz = leftz + 1;


			// leftx and leftz cannot be < 0 and rightx and rightz cannot be >= GRID_WIDTH or GRID_DEPTH
			xRatio = randx - leftx;
			zRatio = randz - leftz;

			p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
		}


		float y1 = heightMap->data[leftx][leftz];
		float y2 = heightMap->data[leftx][rightz];
		float y3 = heightMap->data[rightx][leftz];
		float y4 = heightMap->data[rightx][rightz];

		float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
		float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

		float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;


		particleVertices.push_back(glm::vec3(randx, y, randz));

		stlpSim->mapFromSimulationBox(y);

		p.position = glm::vec3(randx, y, randz);
		p.velocity = glm::vec3(0.0f);

		p.updatePressureVal();

		//particles.push_back(p);
		particleProfiles.push_back(p.profileIndex);
		//particlePressures.push_back(p.pressure);



	}


	//cudaMemcpy(d_particlePressures, &particlePressures[0], sizeof(float) * particlePressures.size(), cudaMemcpyHostToDevice);

	// PARTICLE PROFILES (INDICES) are currently twice on GPU - once in VBO, once in CUDA global memory -> merge!!! (map VBO to CUDA)

	cudaMemcpy(d_profileIndices, &particleProfiles[0], sizeof(int) * particleProfiles.size(), cudaMemcpyHostToDevice);
	glNamedBufferData(particleProfilesVBO, sizeof(int) * particleProfiles.size(), &particleProfiles[0], GL_STATIC_DRAW);


	glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numParticles, particleVertices.data(), GL_STATIC_DRAW);

	cout << numParticles << endl;

	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindVertexArray(0);
	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particleVerticesVBO, cudaGraphicsRegisterFlagsWriteDiscard));

	// unused due to unknown error for now
	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleProfilesVBO, particleProfilesVBO, cudaGraphicsRegisterFlagsReadOnly)); // this is read only for CUDA!



}

void ParticleSystem::initParticlesAboveTerrain() {
	cout << __FUNCTION__ << " not yet implemented!" << endl;
}



//void ParticleSystem::generateParticleOnTerrain(std::vector<glm::vec3>& outVector) {
//
//
//	// testing generation in circle
//	float randx;
//	float randz;
//
//	bool incircle = false;
//	if (incircle) {
//
//		float R = 10.0f;
////		static random_device rd;
//		static mt19937 mt(rd());
//		static uniform_real_distribution<float> dist(0.0f, 1.0f);
//
//		float a = dist(mt) * 2.0f * (float)PI;
//		float r = R * sqrtf(dist(mt));
//
//		randx = r * cos(a);
//		randz = r * sin(a);
//
//		randx += heightMap->width / 2;
//		randz += heightMap->height / 2;
//
//	} else {
//		randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
//		randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));
//	}
//
//	// interpolate
//	int leftx = (int)randx;
//	int rightx = leftx + 1;
//	int leftz = (int)randz;
//	int rightz = leftz + 1;
//
//	// leftx and leftz cannot be < 0 and rightx and rightz cannot be >= GRID_WIDTH or GRID_DEPTH
//	float xRatio = randx - leftx;
//	float zRatio = randz - leftz;
//
//	float y1 = heightMap->data[leftx][leftz];
//	float y2 = heightMap->data[leftx][rightz];
//	float y3 = heightMap->data[rightx][leftz];
//	float y4 = heightMap->data[rightx][rightz];
//
//	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
//	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;
//
//	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;
//
//	//y = 5.0f; //////////////////////////////////////////////////////// FORCE Y to dry adiabat
//
//	particlePositions.push_back(glm::vec3(randx, y, randz));
//
//
//	mapFromSimulationBox(y);
//
//	Particle p;
//	p.position = glm::vec3(randx, y, randz);
//	p.velocity = glm::vec3(0.0f);
//
//
//	if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {
//
//		glm::vec2 p1 = profileMap->data[leftx][leftz];
//		glm::vec2 p2 = profileMap->data[leftx][rightz];
//		glm::vec2 p3 = profileMap->data[rightx][leftz];
//		glm::vec2 p4 = profileMap->data[rightx][rightz];
//
//		glm::vec2 pi1 = zRatio * p2 + (1.0f - zRatio) * p1;
//		glm::vec2 pi2 = zRatio * p4 + (1.0f - zRatio) * p3;
//
//		glm::vec2 pif = xRatio * pi2 + (1.0f - xRatio) * pi1;
//		glm::ivec2 pii = (glm::ivec2)pif;
//
//		if (pii.y != pii.x) {
//			p.profileIndex = (rand() % (pii.y - pii.x) + pii.x) % (stlpDiagram->numProfiles - 1);
//		} else {
//			p.profileIndex = pii.x % (stlpDiagram->numProfiles - 1);
//		}
//
//	} else {
//		p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
//	}
//
//
//	p.updatePressureVal();
//
//	particles.push_back(p);
//	numParticles++;
//
//}





/*

void ParticleSystem::initParticlePositions(int width, int height, bool *collider) {
	cout << "Initializing particle positions." << endl;
	int particleCount = 0;
	float x = 0.0f;
	float y = 0.0f;
	float offset = 0.5f;
	float xOffset = 0.0f;
	float yOffset = 0.0f;
	while (particleCount != numParticles) {
		if (!collider[(int)x + width * (int)y]) {
			particleVertices[particleCount] = glm::vec3(x, y, -1.0f);
			particleCount++;
		}
		y++;
		if (y >= height - 1) {
			y = yOffset;
			x++;
		}
		if (x >= width - 1) {
			yOffset += offset;
			if (yOffset >= 1.0f) {
				yOffset = 0.0f;
				xOffset += offset;
				if (xOffset >= 1.0f) {
					xOffset = 0.0f;
					offset /= 2.0f;
					yOffset += offset;
				}
			}
			x = xOffset;
			y = yOffset;
		}
	}
	cout << "Particle positions intialized!" << endl;
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
}

void ParticleSystem::initParticlePositions(int width, int height, int depth, const HeightMap *hm) {


	// generate in the left wall
	int particleCount = 0;
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float offset = 0.5f;
	float xOffset = 0.0f;
	float yOffset = 0.0f;
	float zOffset = 0.0f;
	while (particleCount != numParticles) {
		if (hm->data[(int)x][(int)z] <= y) {
			particleVertices[particleCount] = glm::vec3(x, y, z);
			particleCount++;
		}
		z++;
		// prefer depth instead of height
		if (z >= depth - 1) {
			z = zOffset;
			y++;
		}
		if (y >= height - 1) {
			y = yOffset;
			z = zOffset;
			x++;
		}
		if (x >= width - 1) {
			xOffset += offset;
			if (xOffset >= 1.0f) {
				xOffset = 0.0f;
				yOffset += offset;
				if (yOffset > 1.0f) {
					yOffset = 0.0f;
					zOffset += offset;
					if (zOffset >= 1.0f) {
						zOffset = 0.0f;
						offset /= 2.0f;
						xOffset += offset;
					}
				}
			}
			x = xOffset;
			y = yOffset;
			z = zOffset;
		}
	}
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
}

void ParticleSystem::copyDataFromVBOtoCPU() {

	printf("Copying data from VBO to CPU in ParticleSystem\n");
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glm::vec3 *tmp = (glm::vec3 *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	for (int i = 0; i < numParticles; i++) {
		particleVertices[i] = tmp[i];
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);


}

*/