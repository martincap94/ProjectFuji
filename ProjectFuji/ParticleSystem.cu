#include "ParticleSystem.h"

#include <cuda_runtime.h>

#include <iostream>

#include "LBM.h"

ParticleSystem::ParticleSystem() {
}

ParticleSystem::ParticleSystem(int numParticles, bool drawStreamlines) : numParticles(numParticles), drawStreamlines(drawStreamlines) {
	particleVertices = new glm::vec3[numParticles]();



	cudaMalloc((void**)&d_numParticles, sizeof(int));

	cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice);


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_STREAM_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glGenBuffers(1, &colorsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	//for (int i = 0; i < numParticles; i++) {
	//	particleVertices[i].z = 1.0f;
	//}
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_STREAM_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


	if (drawStreamlines) {
		streamLines = new glm::vec3[numParticles * MAX_STREAMLINE_LENGTH];

		glGenVertexArrays(1, &streamLinesVAO);
		glBindVertexArray(streamLinesVAO);
		glGenBuffers(1, &streamLinesVBO);
		glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

		glBindVertexArray(0);
	}


	spriteTexture.loadTexture(((string)TEXTURES_DIR + "pointTex.png").c_str());


}


ParticleSystem::~ParticleSystem() {
	delete[] particleVertices;

	if (streamLines != nullptr) {
		delete[] streamLines;
	}
	cudaFree(d_numParticles);
}

void ParticleSystem::draw(const ShaderProgram &shader, bool useCUDA) {

	glUseProgram(shader.id);

	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture.id);

	glPointSize(pointSize);
	shader.setVec3("uColor", particlesColor);

	glBindVertexArray(vao);

	if (!useCUDA) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_STREAM_DRAW);
	}
	//if (lbm->visualizeVelocity) {
	//	glEnableVertexAttribArray(1);
	//} else {
	//	glDisableVertexAttribArray(1);
	//}

	glDrawArrays(GL_POINTS, 0, numParticles);

	if (drawStreamlines) {

		glPointSize(1.0f);
		shader.setVec4("uColor", glm::vec4(0.0f, 0.4f, 1.0f, 1.0f));

		glBindVertexArray(streamLinesVAO);

		glBindBuffer(GL_ARRAY_BUFFER, streamLinesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles * MAX_STREAMLINE_LENGTH, &streamLines[0], GL_STREAM_DRAW);

		glDrawArrays(GL_POINTS, 0, numParticles  * MAX_STREAMLINE_LENGTH);
	}
}

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
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

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
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numParticles, &particleVertices[0], GL_DYNAMIC_DRAW);
}

void ParticleSystem::copyDataFromVBOtoCPU() {

	printf("Copying data from VBO to CPU in ParticleSystem\n");
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glm::vec3 *tmp = (glm::vec3 *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	for (int i = 0; i < numParticles; i++) {
		particleVertices[i] = tmp[i];
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);


}
