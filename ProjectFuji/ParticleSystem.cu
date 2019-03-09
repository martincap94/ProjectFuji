#include "ParticleSystem.h"

#include <cuda_runtime.h>

#include <iostream>

#include "Utils.h"
#include "LBM.h"

ParticleSystem::ParticleSystem(VariableManager *vars) : vars(vars) {

	initBuffers();
	initCUDA();
	spriteTexture.loadTexture(((string)TEXTURES_DIR + "pointTex.png").c_str());

}


ParticleSystem::~ParticleSystem() {
	//delete[] particleVertices;

	cudaFree(d_numParticles);
}



void ParticleSystem::initBuffers() {

	glGenVertexArrays(1, &particleVerticesVAO);
	glBindVertexArray(particleVerticesVAO);
	glGenBuffers(1, &particleVerticesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleVerticesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

}


void ParticleSystem::initCUDA() {

	cudaMalloc((void**)&d_numParticles, sizeof(int));
	cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice);

}


void ParticleSystem::draw(const ShaderProgram &shader, bool useCUDA) {

	glUseProgram(shader.id);

	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture.id);

	glPointSize(pointSize);
	shader.setVec3("u_Color", particlesColor);

	glBindVertexArray(particleVerticesVAO);

	//if (!useCUDA) {
	//	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	//	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_STREAM_DRAW);
	//}
	//if (lbm->visualizeVelocity) {
	//	glEnableVertexAttribArray(1);
	//} else {
	//	glDisableVertexAttribArray(1);
	//}

	glDrawArrays(GL_POINTS, 0, numParticles);

}

void ParticleSystem::initParticlePositionsWithZeros() {
	vector<glm::vec3> particleVertices;

	for (int i = 0; i < numParticles; i++) {
		particleVertices.push_back(glm::vec3(0.0f));
	}

	glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numParticles, particleVertices.data(), GL_STATIC_DRAW);
}

void ParticleSystem::initParticlePositionsOnTerrain() {
}

void ParticleSystem::initParticlePositionsAboveTerrain() {
}





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