#include "StreamlineParticleSystem.h"

#include <cuda_runtime.h>

#include "LBM3D_1D_indices.h"
#include "HeightMap.h"
#include "VariableManager.h"
#include "ShaderManager.h"
#include "CUDAUtils.cuh"


StreamlineParticleSystem::StreamlineParticleSystem(VariableManager *vars) : vars(vars) {
	
	// preset default values but DO NOT initialize!
	this->maxNumStreamlines = vars->maxNumStreamlines;
	this->maxStreamlineLength = vars->maxStreamlineLength;

	shader = ShaderManager::getShaderPtr("singleColor");

}


StreamlineParticleSystem::~StreamlineParticleSystem() {
	//if (streamlineVBOs) {
	//	delete[] streamlineVBOs;
	//}
	if (currActiveVertices) {
		delete[] currActiveVertices;
	}
	if (cudaStreamlinesVBO) {
		CHECK_ERROR(cudaGraphicsUnregisterResource(cudaStreamlinesVBO));
	}
}

void StreamlineParticleSystem::draw() {
	if (!initialized) {
		return;
	}
	if (visible) {

		shader->use();
		shader->setColor(glm::vec3(0.5f, 1.0f, 0.8f));

		glPointSize(4.0f);


		glBindVertexArray(streamlinesVAO);

		//cout << " (drawing) max num streamlines = " << maxNumStreamlines << endl;

		for (int i = 0; i < maxNumStreamlines; i++) {
			if (liveLineCleanup || !active) {
				glDrawArrays(GL_LINE_STRIP, i * maxStreamlineLength, currActiveVertices[i] + 1);
				glDrawArrays(GL_POINTS, i * maxStreamlineLength + currActiveVertices[i], 1);

			} else {
				glDrawArrays(GL_LINE_STRIP, i * maxStreamlineLength, maxStreamlineLength);
			}

		}
		//glDrawArrays(GL_LINE_STRIP, 0, maxStreamlineLength * maxNumStreamlines);



		/*
		for (int i = 0; i < maxNumStreamlines; i++) {
			glBindVertexBuffer(0, streamlineVBOs[i], 0, sizeof(glm::vec3));
			glDrawArrays(GL_LINES, 0, maxStreamlineLength); // TO DO - specify actual amount of vertices to be drawn (not maximum)
		}
		*/



		glBindVertexArray(0);
	}
}

void StreamlineParticleSystem::init() {
	initBuffers();
	initCUDA();
	initialized = true;
}

void StreamlineParticleSystem::update() {
	if (liveLineCleanup) {
		cleanupLines();
	}
}

void StreamlineParticleSystem::initBuffers() {

	glGenVertexArrays(1, &streamlinesVAO);
	glGenBuffers(1, &streamlinesVBO);

	glBindVertexArray(streamlinesVAO);
	glBindBuffer(GL_ARRAY_BUFFER, streamlinesVBO);

	//glm::vec3 *initVertexData = new glm::vec3[maxStreamlineLength * maxNumStreamlines]();

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * maxStreamlineLength * maxNumStreamlines, NULL, GL_DYNAMIC_DRAW);
	//GLuint zero = 0;
	//glClearBufferData(GL_ARRAY_BUFFER, GL_RGB32F, GL_R32F, GL_UNSIGNED_INT, &zero);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);


	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaStreamlinesVBO, streamlinesVBO, cudaGraphicsMapFlagsWriteDiscard));

	currActiveVertices = new int[maxNumStreamlines]();
	//for (int i = 0; i < maxNumStreamlines; i++) {
	//	currActiveVertices[i] = 0;
	//}

	//delete[] initVertexData;

	/*
	glGenVertexArrays(1, &streamlinesVAO);

	glBindVertexArray(streamlinesVAO);
	glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
	glVertexAttribBinding(0, 0);
	glBindVertexArray(0);

	glm::vec3 *initVertexData = new glm::vec3[maxStreamlineLength]();
	streamlineVBOs = new GLuint[maxNumStreamlines]();

	glGenBuffers(maxNumStreamlines, &streamlineVBOs[0]);
	for (int i = 0; i < maxNumStreamlines; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, streamlineVBOs[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(initVertexData), &initVertexData, GL_DYNAMIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	delete[] initVertexData;
	*/



}

void StreamlineParticleSystem::initCUDA() {

	CHECK_ERROR(cudaMalloc(&d_currActiveVertices, sizeof(int) * maxNumStreamlines));
	CHECK_ERROR(cudaMemset(d_currActiveVertices, 0, sizeof(int) * maxNumStreamlines));
}

void StreamlineParticleSystem::activate() {
	active = true;
}

void StreamlineParticleSystem::deactivate() {
	active = false;

	// no need to cleanup if we were cleaning up each frame
	if (!liveLineCleanup) {
		cleanupLines();
	}


}

void StreamlineParticleSystem::reset() {
	// maybe this could work for both active and inactive particles (as online/offline reset)
	CHECK_ERROR(cudaMemset(d_currActiveVertices, 0, sizeof(int) * maxNumStreamlines));
	frameCounter = 0;



}

void StreamlineParticleSystem::cleanupLines() {
	// copy back the amounts of vertices that create individual lines (so we can get rid off the unwanted lines)
	CHECK_ERROR(cudaMemcpy(currActiveVertices, d_currActiveVertices, sizeof(int) * maxNumStreamlines, cudaMemcpyDeviceToHost));
}

void StreamlineParticleSystem::setPositionInHorizontalLine() {
	if (active) {
		reset();
	}

	// quick testing
	float zoffset = lbm->position.z;
	float zstep = lbm->getWorldDepth() / (float)maxNumStreamlines;
	float x = lbm->position.x + 1.0f;
	float y = lbm->position.y + lbm->getWorldHeight() / 2.0f; // TO DO - add LBM mid points (in LBM class)

	glBindBuffer(GL_ARRAY_BUFFER, streamlinesVBO);
	for (int i = 0; i < maxNumStreamlines; i++) {
		glm::vec3 pos(x, y, i * zstep + zoffset);
		glBufferSubData(GL_ARRAY_BUFFER, i * maxStreamlineLength * sizeof(glm::vec3), sizeof(glm::vec3), glm::value_ptr(pos));
	}

}

void StreamlineParticleSystem::setPositionInVerticalLine() {
	if (active) {
		reset();
	}

	// quick testing
	float yoffset = lbm->position.y;
	float ystep = (lbm->getWorldHeight() - FLT_EPSILON) / (float)maxNumStreamlines;
	float x = lbm->position.x + FLT_EPSILON;
	float z = lbm->position.z + lbm->getWorldDepth() / 2.0f;

	glBindBuffer(GL_ARRAY_BUFFER, streamlinesVBO);
	for (int i = 0; i < maxNumStreamlines; i++) {
		glm::vec3 pos(x, i * ystep + yoffset, z);
		glBufferSubData(GL_ARRAY_BUFFER, i * maxStreamlineLength * sizeof(glm::vec3), sizeof(glm::vec3), glm::value_ptr(pos));
	}
}

void StreamlineParticleSystem::setPositionCross() {



}
