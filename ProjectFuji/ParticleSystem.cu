#include "ParticleSystem.h"

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "Utils.h"
#include "LBM.h"
#include "CUDAUtils.cuh"

#include "Emitter.h"
#include "PositionalEmitter.h"
#include "CircleEmitter.h"
//#include "CDFEmitter.h"
#include "CDFEmitter.h"
#include "EmitterBrushMode.h"


#include "TextureManager.h"

#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include <thrust\execution_policy.h>
#include <thrust\sequence.h>

#include <stdio.h>

#include <filesystem>
namespace fs = std::experimental::filesystem;



__global__ void computeParticleDistances(glm::vec3 *particleVertices, float *particleDistances, glm::vec3 referencePosition, int numParticles) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {

		particleDistances[idx] = glm::distance(particleVertices[idx], referencePosition);

	}

}

__global__ void computeParticleProjectedDistances(glm::vec3 *particleVertices, float *particleDistances, glm::vec3 sortVector, int numParticles) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {
		particleDistances[idx] = glm::dot(particleVertices[idx], sortVector);
	}


}


__global__ void checkParticleValidityKernel(glm::vec3 *particleVertices, int numParticles) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {
		glm::vec3 pos = particleVertices[idx];
		if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z) || isinf(pos.x) || isinf(pos.y) || isinf(pos.z)) {
			//printf("oh no");
			particleVertices[idx] = glm::vec3(0.0f);
		}

		
		/*
		// DO NOT USE THIS - it is much more readable, but it doesn't seem to work correctly on GPU!
		if (glm::any(glm::isnan(pos)) || glm::any(glm::isinf(pos))) {
			particleVertices[idx] = glm::vec3(0.0f);
		}
		*/

	}
}

__global__ void clearVerticalVelocitiesKernel(float *verticalVelocities, int numParticles) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < numParticles) {
		verticalVelocities[idx] = 0.0f;
	}
}



ParticleSystem::ParticleSystem(VariableManager *vars) : vars(vars) {

	loadParticleSaveFiles();

	curveShader = ShaderManager::getShaderPtr("curve");
	pointSpriteTestShader = ShaderManager::getShaderPtr("pointSpriteTest");
	singleColorShader = ShaderManager::getShaderPtr("singleColor");


	heightMap = vars->heightMap;
	numParticles = vars->numParticles;

	blockDim = dim3(256, 1, 1);
	gridDim = dim3((int)ceil((float)numParticles / (float)blockDim.x), 1, 1);


	numActiveParticles = 0;
	//numActiveParticles = numParticles;

	initBuffers();
	initCUDA();

	//spriteTexture = TextureManager::loadTexture((string)TEXTURES_DIR + "radial-gradient-white-2.png");
	//secondarySpriteTexture = TextureManager::loadTexture((string)TEXTURES_DIR + "radial-gradient-white-2.png");

	spriteTexture = TextureManager::loadTexture((string)TEXTURES_DIR + "testTexture.png");
	secondarySpriteTexture = TextureManager::loadTexture((string)TEXTURES_DIR + "testTexture2.png");


	//spriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture.png").c_str());
	//secondarySpriteTexture.loadTexture(((string)TEXTURES_DIR + "testTexture2.png").c_str());


	disableAllEmitters();

	//formBoxVisModel = new Model("models/unitbox.fbx");
	formBoxVisShader = ShaderManager::getShaderPtr("singleColorModel");
	formBoxVisModel = new Model("models/unitbox.fbx");



}


ParticleSystem::~ParticleSystem() {
	//delete[] particleVertices;

	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaParticleVerticesVBO));
	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaParticleProfilesVBO));
	CHECK_ERROR(cudaGraphicsUnregisterResource(cudaDiagramParticleVerticesVBO));

	for (int i = 0; i < emitters.size(); i++) {
		delete emitters[i];
	}

	cudaFree(d_numParticles);

	if (formBoxVisModel) {
		delete formBoxVisModel;
	}

}


void ParticleSystem::update() {
	if (ebm->isActive()) {
		if (ebm->hasActiveBrush()) {
			ebm->getActiveBrushPtr()->update();
		}
	} else {
		for (int i = 0; i < emitters.size(); i++) {
			emitters[i]->update();
		}
	}
	emitParticles();
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

	vector<unsigned int> indices;
	for (int i = 0; i < numParticles; i++) {
		indices.push_back(i);
	}

	glGenBuffers(1, &particlesEBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, particlesEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numParticles * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);

	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticlesEBO, particlesEBO, cudaGraphicsMapFlagsWriteDiscard));




	///////////////////////////////////////////////////////////////////////////////////////
	// DIAGRAM
	///////////////////////////////////////////////////////////////////////////////////////
	glGenVertexArrays(1, &diagramParticlesVAO);
	glBindVertexArray(diagramParticlesVAO);
	glGenBuffers(1, &diagramParticleVerticesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, diagramParticleVerticesVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (void *)0);

	glBindVertexArray(0);

}


void ParticleSystem::initCUDA() {

	CHECK_ERROR(cudaMalloc((void**)&d_numParticles, sizeof(int)));
	CHECK_ERROR(cudaMemcpy(d_numParticles, &numParticles, sizeof(int), cudaMemcpyHostToDevice));

	CHECK_ERROR(cudaMalloc((void**)&d_verticalVelocities, sizeof(float) * numParticles));

	CHECK_ERROR(cudaMemset(d_verticalVelocities, 0, sizeof(float) * numParticles));

	//cudaGLRegisterBufferObject(cudaDiagramParticleVerticesVBO, )

	CHECK_ERROR(cudaMalloc((void**)&d_particleDistances, sizeof(float) * numParticles));
	CHECK_ERROR(cudaMemset(d_particleDistances, 0, sizeof(float) * numParticles));


}

void ParticleSystem::emitParticles() {

	//// check if emitting particles is possible (maximum reached)
	// --> checking is also done in each emitter, this prevents further unnecessary work
	if (numActiveParticles >= numParticles) {
		return;
	}

	int prevNumActiveParticles = numActiveParticles;


	if (ebm->isActive()) {
		if (ebm->hasActiveBrush()) {
			ebm->getActiveBrushPtr()->emitParticles(ebm->numParticlesEmittedPerFrame);
		}
	} else {

		// go through all emitters and emit particles (each pushes them to this system)
		for (int i = 0; i < emitters.size(); i++) {
			emitters[i]->emitParticles();
		}
	}

	//cout << "num particles to upload = " << particleVerticesToEmit.size() << endl;

	// upload the data to VBOs and CUDA memory

	glNamedBufferSubData(particleVerticesVBO, sizeof(glm::vec3) * prevNumActiveParticles, sizeof(glm::vec3) * particleVerticesToEmit.size()/*(numActiveParticles - prevNumActiveParticles)*/, particleVerticesToEmit.data());

	glNamedBufferSubData(particleProfilesVBO, sizeof(int) * prevNumActiveParticles, sizeof(int) * particleProfilesToEmit.size(), particleProfilesToEmit.data());

	//cout << verticalVelocitiesToEmit.size() << endl;
	//cout << " | prevNumActiveParticles = " << prevNumActiveParticles << endl;
	//cout << " | numActiveParticles     = " << numActiveParticles << endl;
	//cout << " | active - prevActive    = " << (numActiveParticles - prevNumActiveParticles) << endl;

	CHECK_ERROR(cudaMemcpy(d_verticalVelocities + prevNumActiveParticles, verticalVelocitiesToEmit.data(), verticalVelocitiesToEmit.size() * sizeof(float), cudaMemcpyHostToDevice));


	// clear the temporary emitted particle structures

	particleVerticesToEmit.clear();
	particleProfilesToEmit.clear();
	verticalVelocitiesToEmit.clear();

}





// NOT USED ANYMORE
void ParticleSystem::draw(glm::vec3 cameraPos) {


	/*
	size_t num_bytes;
	glm::vec3 *d_mappedParticleVerticesVBO;

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticleVerticesVBO, &num_bytes, cudaParticleVerticesVBO));

	CHECK_ERROR(cudaGetLastError());

	computeParticleDistances << <gridDim.x, blockDim.x >> > (d_mappedParticleVerticesVBO, d_particleDistances, cameraPos, numActiveParticles);

	CHECK_ERROR(cudaGetLastError());

	// this is a no go, we need to sort by key! -> more memory...
	//thrust::device_ptr<glm::vec3> dptr(d_mappedParticleVerticesVBO);  // add this line before the sort line
	//thrust::sort(dptr, dptr + numActiveParticles);        // modify your sort line


	//thrust::sort_by_key(keys.begin(), keys.end(), values.begin());


	//// OLD APPROACH
	//thrust::device_ptr<glm::vec3> thrustParticleVerticesPtr(d_mappedParticleVerticesVBO);
	//thrust::device_ptr<float> thrustParticleDistancesPtr(d_particleDistances);

	//thrust::sort_by_key(thrustParticleDistancesPtr, thrustParticleDistancesPtr + numActiveParticles, thrustParticleVerticesPtr, thrust::greater<float>());


	// NEW APPROACH
	thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater<float>());
	CHECK_ERROR(cudaGetLastError());


	//cudaDeviceSynchronize(); // if we do not synchronize, thrust will (?) throw a system error since we unmap the resource before it finishes sorting
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0));
	*/

	ShaderProgram *shader;
	if (vars->usePointSprites) {

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glDepthMask(GL_FALSE);
		shader = pointSpriteTestShader;

	} else {
		shader = singleColorShader;
	}

	//glUseProgram(point.id);
	pointSpriteTestShader->use();

	pointSpriteTestShader->setBool("u_ShowHiddenParticles", (bool)showHiddenParticles);
	pointSpriteTestShader->setInt("u_Tex", 0);
	pointSpriteTestShader->setInt("u_SecondTex", 1);
	pointSpriteTestShader->setVec3("u_TintColor", vars->tintColor);

	pointSpriteTestShader->setInt("u_OpacityBlendMode", opacityBlendMode);
	pointSpriteTestShader->setFloat("u_OpacityBlendRange", opacityBlendRange);


	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, spriteTexture->id);

	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, secondarySpriteTexture->id);

	glPointSize(pointSize);
	pointSpriteTestShader->setVec3("u_CameraPos", cameraPos);
	pointSpriteTestShader->setFloat("u_PointSizeModifier", pointSize);
	pointSpriteTestShader->setFloat("u_OpacityMultiplier", vars->opacityMultiplier);

	glBindVertexArray(particlesVAO);


	//glDrawArrays(GL_POINTS, 0, numActiveParticles);
	glDrawElements(GL_POINTS, numActiveParticles, GL_UNSIGNED_INT, 0);


	glDepthMask(GL_TRUE);



	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->draw();
	}



}

void ParticleSystem::drawGeometry(ShaderProgram *shader, glm::vec3 cameraPos) {

	shader->use();

	glPointSize(pointSize);
	shader->setModelMatrix(glm::mat4(1.0));
	shader->setVec3("u_CameraPos", cameraPos);
	shader->setFloat("u_PointSizeModifier", pointSize);
	shader->setBool("u_IsInstanced", false);

	glBindVertexArray(particlesVAO);
	glDrawArrays(GL_POINTS, 0, numActiveParticles);

}



void ParticleSystem::drawDiagramParticles() {
	curveShader->use();
	GLboolean depthTestEnabled;
	glGetBooleanv(GL_DEPTH_TEST, &depthTestEnabled);
	glDisable(GL_DEPTH_TEST);


	glPointSize(2.0f);
	curveShader->setColor(diagramParticlesColor);

	glBindVertexArray(diagramParticlesVAO);
	//glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * particlePoints.size(), &particlePoints[0], GL_DYNAMIC_DRAW);
	//glNamedBufferData(particlesVBO, sizeof(glm::vec2) * particlePoints.size(), &particlePoints[0], GL_DYNAMIC_DRAW);
	glDrawArrays(GL_POINTS, 0, numDiagramParticlesToDraw);

	if (depthTestEnabled) {
		glEnable(GL_DEPTH_TEST);
	}

}

void ParticleSystem::drawHelperStructures() {
	if (editingFormBox) {
		formBoxVisShader->use();
		formBoxVisShader->setColor(glm::vec3(1.0f, 0.0f, 0.0f));
		formBoxVisModel->transform.position = newFormBoxSettings.position;
		formBoxVisModel->transform.scale = newFormBoxSettings.size;
		formBoxVisModel->update();
		formBoxVisModel->drawWireframe(formBoxVisShader);
	}


	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->draw();
	}


}


void ParticleSystem::sortParticlesByDistance(glm::vec3 referencePoint, eSortPolicy sortPolicy) {


	size_t num_bytes;
	glm::vec3 *d_mappedParticleVerticesVBO;
	unsigned int *d_mappedParticlesEBO;

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticleVerticesVBO, &num_bytes, cudaParticleVerticesVBO));

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticlesEBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticlesEBO, &num_bytes, cudaParticlesEBO));


	CHECK_ERROR(cudaGetLastError());

	computeParticleDistances << <gridDim.x, blockDim.x >> > (d_mappedParticleVerticesVBO, d_particleDistances, referencePoint, numActiveParticles);

	CHECK_ERROR(cudaGetLastError());

	thrust::sequence(thrust::device, d_mappedParticlesEBO, d_mappedParticlesEBO + numActiveParticles);

	switch (sortPolicy) {
		case GREATER:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::greater<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater<float>());
			break;
		case LESS:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::less<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less<float>());
			break;
		case GEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::greater_equal<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater_equal<float>());
			break;
		case LEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::less_equal<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less_equal<float>());
			break;
	}

	CHECK_ERROR(cudaGetLastError());


	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticlesEBO, 0));

	/*
	size_t num_bytes;
	glm::vec3 *d_mappedParticleVerticesVBO;

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticleVerticesVBO, &num_bytes, cudaParticleVerticesVBO));

	CHECK_ERROR(cudaGetLastError());

	computeParticleDistances << <gridDim.x, blockDim.x >> > (d_mappedParticleVerticesVBO, d_particleDistances, referencePoint, numActiveParticles);

	CHECK_ERROR(cudaGetLastError());

	switch (sortPolicy) {
		case GREATER:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater<float>());
			break;
		case LESS:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less<float>());
			break;
		case GEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater_equal<float>());
			break;
		case LEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less_equal<float>());
			break;
	}

	CHECK_ERROR(cudaGetLastError());


	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0));

	*/
}

void ParticleSystem::sortParticlesByProjection(glm::vec3 sortVector, eSortPolicy sortPolicy) {


	glm::vec3 *d_mappedParticleVerticesVBO;
	unsigned int *d_mappedParticlesEBO;

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticleVerticesVBO, nullptr, cudaParticleVerticesVBO));

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticlesEBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticlesEBO, nullptr, cudaParticlesEBO));


	CHECK_ERROR(cudaGetLastError());

	computeParticleProjectedDistances << <gridDim.x, blockDim.x >> > (d_mappedParticleVerticesVBO, d_particleDistances, sortVector, numActiveParticles);

	CHECK_ERROR(cudaGetLastError());

	thrust::sequence(thrust::device, d_mappedParticlesEBO, d_mappedParticlesEBO + numActiveParticles);



	switch (sortPolicy) {
		case GREATER:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::greater<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater<float>());
			break;
		case LESS:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::less<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less<float>());
			break;
		case GEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::greater_equal<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::greater_equal<float>());
			break;
		case LEQUAL:
			thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO, thrust::less_equal<float>());
			//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticleVerticesVBO, thrust::less_equal<float>());
			break;
	}

	//thrust::sort_by_key(thrust::device, d_particleDistances, d_particleDistances + numActiveParticles, d_mappedParticlesEBO);

	CHECK_ERROR(cudaGetLastError());


	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticlesEBO, 0));



}

void ParticleSystem::checkParticleValidity() {


	glm::vec3 *d_mappedParticleVerticesVBO;

	CHECK_ERROR(cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0));
	CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&d_mappedParticleVerticesVBO, nullptr, cudaParticleVerticesVBO));

	checkParticleValidityKernel << <gridDim.x, blockDim.x >> > (d_mappedParticleVerticesVBO, numActiveParticles);
	
	CHECK_ERROR(cudaGetLastError());

	CHECK_ERROR(cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0));

}

void ParticleSystem::initParticlesWithZeros() {
	cout << __FUNCTION__ << " not yet implemented!" << endl;

}

void ParticleSystem::initParticlesOnTerrain() {

	refreshParticlesOnTerrain();

	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particleVerticesVBO, cudaGraphicsRegisterFlagsWriteDiscard));
	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaParticleProfilesVBO, particleProfilesVBO, cudaGraphicsRegisterFlagsReadOnly)); // this is read only for CUDA!
	CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&cudaDiagramParticleVerticesVBO, diagramParticleVerticesVBO, cudaGraphicsRegisterFlagsWriteDiscard));

}

void ParticleSystem::initParticlesAboveTerrain() {
	cout << __FUNCTION__ << " not yet implemented!" << endl;
}

void ParticleSystem::formBox() {
	formBox(formBoxSettings.position, formBoxSettings.size);
}

void ParticleSystem::formBox(glm::vec3 pos, glm::vec3 size) {

	vector<glm::vec3> particleVertices;
	for (int i = 0; i < numParticles; i++) {
		particleVertices.push_back(glm::vec3(getRandFloat(pos.x, pos.x + size.x), getRandFloat(pos.y, pos.y + size.y), getRandFloat(pos.z, pos.z + size.z)));
	}



	glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numParticles, particleVertices.data(), GL_STATIC_DRAW);
	clearVerticalVelocities();
}

void ParticleSystem::refreshParticlesOnTerrain() {

	vector<glm::vec3> particleVertices;
	vector<int> particleProfiles;
	vector<float> particlePressures;
	vector<glm::vec2> diagramParticleVertices;

	ppmImage *profileMap = stlpSim->profileMap;
	STLPDiagram *stlpDiagram = stlpSim->stlpDiagram;

	for (int i = 0; i < numParticles; i++) {
		Particle p;
		p.position = heightMap->getRandomWorldPosition();


		glm::ivec3 texelPos = p.position / heightMap->texelWorldSize;

		if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {
			p.profileIndex = (rand() % (texelPos.y - texelPos.x) + texelPos.x) % (stlpDiagram->numProfiles - 1);
		} else {
			p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
		}

		p.updatePressureVal();

		float normP = stlpDiagram->getNormalizedPres(p.pressure);
		glm::vec2 dryAdiabatIntersection = stlpDiagram->dryAdiabatProfiles[p.profileIndex].getIntersectionWithIsobar(normP);
		float particleTemp = stlpDiagram->getDenormalizedTemp(dryAdiabatIntersection.x, normP);

		particleVertices.push_back(p.position);
		particleProfiles.push_back(p.profileIndex);
		diagramParticleVertices.push_back(stlpDiagram->getNormalizedCoords(particleTemp, p.pressure));




		/*

		// LEGACY IMPLEMENTATION


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


		float y1 = heightMap->data[leftx + leftz * heightMap->width];
		float y2 = heightMap->data[leftx + rightz * heightMap->width];
		float y3 = heightMap->data[rightx + leftz * heightMap->width];
		float y4 = heightMap->data[rightx + rightz * heightMap->width];

		float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
		float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

		float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;


		particleVertices.push_back(glm::vec3(randx, y, randz));

		//stlpSim->mapFromSimulationBox(y);

		p.position = glm::vec3(randx, y, randz);
		p.velocity = glm::vec3(0.0f);

		p.updatePressureVal();

		//float particleTemp = stlpDiagram->getDenormalizedTemp(dryAdiabatIntersection.x, normP);

		float normP = stlpDiagram->getNormalizedPres(p.pressure);
		glm::vec2 dryAdiabatIntersection = stlpDiagram->dryAdiabatProfiles[p.profileIndex].getIntersectionWithIsobar(normP);
		float particleTemp = stlpDiagram->getDenormalizedTemp(dryAdiabatIntersection.x, normP);

		diagramParticleVertices.push_back(stlpDiagram->getNormalizedCoords(particleTemp, p.pressure));

		//particles.push_back(p);
		particleProfiles.push_back(p.profileIndex);

		//particlePressures.push_back(p.pressure);
		*/


	}

	glNamedBufferData(particleProfilesVBO, sizeof(int) * particleProfiles.size(), &particleProfiles[0], GL_STATIC_DRAW);


	glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numParticles, particleVertices.data(), GL_STATIC_DRAW);

	glNamedBufferData(diagramParticleVerticesVBO, sizeof(glm::vec2) * numParticles, diagramParticleVertices.data(), GL_STATIC_DRAW);

}

void ParticleSystem::clearVerticalVelocities(bool clearActiveOnly) {
	clearVerticalVelocitiesKernel << <gridDim.x, blockDim.x >> > (d_verticalVelocities, clearActiveOnly ? numActiveParticles : numParticles);
}

void ParticleSystem::activateAllParticles() {
	numActiveParticles = numParticles;
}

void ParticleSystem::deactivateAllParticles() {
	numActiveParticles = 0;
}

void ParticleSystem::activateAllDiagramParticles() {
	numDiagramParticlesToDraw = numActiveParticles;
}

void ParticleSystem::deactivateAllDiagramParticles() {
	numDiagramParticlesToDraw = 0;
}

void ParticleSystem::enableAllEmitters() {
	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->enabled = true;
	}
}

void ParticleSystem::disableAllEmitters() {
	for (int i = 0; i < emitters.size(); i++) {
		emitters[i]->enabled = false;
	}
}

void ParticleSystem::createPredefinedEmitters() {

	//ech.circleEmitter = new CircleEmitter();
	//ech.cdfEmitter = new CDFEmitter();


	emitters.push_back(new CircleEmitter("Circle test", this, glm::vec3(4000.0f, 0.0f, 4000.0f), 2000.0f));
	emitters.push_back(new CDFEmitter("CDF test", this, "textures/cdf2.png"));
	emitters.push_back(new PositionalCDFEmitter("Positional CDF test", this, "icons/edit.png"));
	emitters.push_back(new CDFEmitter("CDF Dynamic", this, "textures/cdf2.png", true));
	emitters.push_back(new PositionalCDFEmitter("DCGI", this, "textures/dcgi_cdf_mirrored.png"));
	TextureManager::loadTexture("textures/dcgi_cdf.png");
}

void ParticleSystem::createEmitter(int emitterType, string emitterName) {
	Emitter *createdEmitter = nullptr;

	switch (emitterType) {
		case Emitter::eEmitterType::CIRCULAR: {
			createdEmitter = new CircleEmitter(ech.circleEmitter, this);
			break;
		}
		case Emitter::eEmitterType::CDF_TERRAIN: {
			createdEmitter = new CDFEmitter(ech.cdfEmitter, this);
			break;
		}
		case Emitter::eEmitterType::CDF_POSITIONAL: {
			createdEmitter = new PositionalCDFEmitter(ech.pcdfEmitter, this);
			break;
		}
		default:
			break;
	}

	if (createdEmitter != nullptr) {
		createdEmitter->name = emitterName;
		emitters.push_back(createdEmitter);
	}


}

void ParticleSystem::deleteEmitter(int idx) {
	if (idx >= emitters.size()) {
		cout << "Cannot delete emitter at idx " << idx << ": it is out of bounds!" << endl;
		return;
	}
	if (emitters[idx]) {
		delete emitters[idx];
	}
	emitters.erase(emitters.begin() + idx);


}


// Do not use this, it does not work with the CUDA compiler
void ParticleSystem::constructEmitterCreationWindow(nk_context * ctx, UserInterface * ui, int emitterType, bool &closeWindowAfterwards) {
	nk_layout_row_dynamic(ctx, 30, 1);

	static char nameBuffer[64];
	static int nameLength;
	nk_flags event = nk_edit_string(ctx, NK_EDIT_SIMPLE, &nameBuffer[0], &nameLength, 64, nk_filter_default);

	if (event & NK_EDIT_ACTIVATED) {
		vars->generalKeyboardInputEnabled = false;
	}
	if (event & NK_EDIT_DEACTIVATED) {
		vars->generalKeyboardInputEnabled = true;
	}
	nameBuffer[nameLength] = '\0';
	string eName = string(nameBuffer);
	cout << "|" << eName << "|" << endl;
	nk_layout_row_dynamic(ctx, 15, 1);

	switch (emitterType) {
		case Emitter::eEmitterType::CIRCULAR: {
			ech.circleEmitter.constructEmitterPropertiesTab(ctx, ui);
			break;
		}
		case Emitter::eEmitterType::CDF_TERRAIN: {
			ech.cdfEmitter.constructEmitterPropertiesTab(ctx, ui);
			break;
		}
		case Emitter::eEmitterType::CDF_POSITIONAL: {
			ech.pcdfEmitter.constructEmitterPropertiesTab(ctx, ui);
			break;
		}
		default:
			break;

	}

	nk_layout_row_dynamic(ctx, 15, 1);

	if (nk_button_label(ctx, "Create Emitter")) {
		createEmitter(emitterType, eName);
	}
	if (nk_button_label(ctx, "Create and Close")) {
		createEmitter(emitterType, eName);
		closeWindowAfterwards = true;
	}
	if (nk_button_label(ctx, "Close")) {
		closeWindowAfterwards = true;
	}




}

void ParticleSystem::pushParticleToEmit(Particle p) {
	particleVerticesToEmit.push_back(p.position);
	particleProfilesToEmit.push_back(p.profileIndex);
	verticalVelocitiesToEmit.push_back(p.velocity.y);
	numActiveParticles++; // each emitter already checks if numActiveParticles < numParticles, no need to check once more

}

void ParticleSystem::saveParticlesToFile(std::string filename, bool saveOnlyActive) {

	if (!fs::exists(PARTICLE_DATA_DIR)) {
		fs::create_directory(PARTICLE_DATA_DIR);

	} else {
		if (!fs::is_directory(PARTICLE_DATA_DIR)) {
			cout << "Cannot save particles! Please make sure " << PARTICLE_DATA_DIR << " is a directory!" << endl;
			return;
		}
	}

	string fullFilename = PARTICLE_DATA_DIR + filename + ".bin";
	if (fs::exists(fullFilename)) {
		cout << "File " << fullFilename << " exists, will be rewritten!" << endl;
	}



	glm::vec3 *vertexData = (glm::vec3 *)glMapNamedBuffer(particleVerticesVBO, GL_READ_ONLY);
	int *profileData = (int *)glMapNamedBuffer(particleProfilesVBO, GL_READ_ONLY);


	
	ofstream out(fullFilename, ios::binary | ios::out);
	//out << numParticles << endl;
	//out << numActiveParticles << endl;

	int numParticlesToSave = numParticles;
	if (saveOnlyActive) {
		numParticlesToSave = numActiveParticles;
	}

	out.write((char *)&numParticlesToSave, sizeof(int));
	out.write((char *)&numActiveParticles, sizeof(int));

	out.write((char *)&vertexData[0], numParticlesToSave * sizeof(glm::vec3));
	out.write((char *)&profileData[0], numParticlesToSave * sizeof(int));

	//for (int i = 0; i < numParticles; i++) {
	//	out << vertexData[i].x << ' ' << vertexData[i].y << ' ' << vertexData[i].z << ' ' << profileData[i] << endl;
	//}
	//

	glUnmapNamedBuffer(particleVerticesVBO);
	glUnmapNamedBuffer(particleProfilesVBO);



}

void ParticleSystem::constructSaveParticlesWindow(nk_context * ctx, UserInterface * ui, bool & closeWindowAfterwards) {

	static string particleSaveName;
	const static int bufferLength = 32;
	static char nameBuffer[bufferLength];
	static int nameLength;
	static int saveActiveParticlesOnly = 0;

	ui->nk_property_string(ctx, particleSaveName, nameBuffer, bufferLength, nameLength);
	nk_checkbox_label(ctx, "Save Active Particles Only", &saveActiveParticlesOnly);

	if (nameLength == 0) {
		ui->setButtonStyle(ctx, false);
		nk_button_label(ctx, "Save");
		nk_button_label(ctx, "Save and Close");
		ui->setButtonStyle(ctx, true);
	} else {
		if (nk_button_label(ctx, "Save")) {
			saveParticlesToFile(particleSaveName, saveActiveParticlesOnly);
		}
		if (nk_button_label(ctx, "Save and Close")) {
			saveParticlesToFile(particleSaveName, saveActiveParticlesOnly);
			closeWindowAfterwards = true;
		}
	}
	if (nk_button_label(ctx, "Close")) {
		closeWindowAfterwards = true;
	}

}

void ParticleSystem::constructLoadParticlesWindow(nk_context * ctx, UserInterface * ui, bool & closeWindowAfterwards) {
	
	static bool fileSelected = false;
	static string selectedFile;

	nk_layout_row_dynamic(ctx, 15, 1);
	if (nk_combo_begin_label(ctx, fileSelected ? selectedFile.c_str() : "Select file...", nk_vec2(nk_widget_width(ctx), 200))) {
		nk_layout_row_dynamic(ctx, 15, 1);
		if (nk_combo_item_label(ctx, "None", NK_TEXT_LEFT)) {
			fileSelected = false;
			nk_combo_close(ctx);
		}
		for (int i = 0; i < particleSaveFiles.size(); i++) {
			if (nk_combo_item_label(ctx, particleSaveFiles[i].c_str(), NK_TEXT_LEFT)) {
				selectedFile = particleSaveFiles[i];
				fileSelected = true;
				nk_combo_close(ctx);
			}
		}
		nk_combo_end(ctx);
	}

	if (!fileSelected) {
		ui->setButtonStyle(ctx, false);
		nk_button_label(ctx, "Load");
		ui->setButtonStyle(ctx, true);
	} else {
		if (nk_button_label(ctx, "Load")) {
			loadParticlesFromFile(selectedFile);
		}
	}
	if (nk_button_label(ctx, "Close")) {
		closeWindowAfterwards = true;
	}

}

void ParticleSystem::loadParticlesFromFile(std::string filename) {
	
	if (!fs::exists(filename) || !fs::is_regular_file(filename)) {
		cout << "Particle file " << filename << " could not be loaded!" << endl;
		return;
	}

	ifstream infile(filename, ios::binary | ios::in);
	int inNumParticles;
	int inNumActiveParticles;
	//infile >> inNumParticles;
	//infile >> inNumActiveParticles;
	infile.read((char *)&inNumParticles, sizeof(int));
	infile.read((char *)&inNumActiveParticles, sizeof(int));


	if (inNumActiveParticles > inNumParticles) {
		printf("There is something wrong with the loaded file: numParticles (%d) < numActiveParticles (%d)!\n", inNumParticles, inNumActiveParticles);
		return;
	} else if (inNumActiveParticles <= 0 || inNumParticles <= 0) {
		printf("There is something wrong with the loaded file: numParticles (%d) <= 0 || numActiveParticles (%d) <= 0!\n", inNumParticles, inNumActiveParticles);
		return;
	}

	bool bufferSubData = false;
	if (inNumParticles > numParticles) {
		cout << "We do not support loading more particles than the application was configured with!" << endl;
		cout << " | Current configuration = " << numParticles << endl;
		cout << " | Loaded configuration  = " << inNumActiveParticles << endl;
		cout << " | ---> Only " << numParticles << " will be loaded instead..." << endl;
		inNumParticles = numParticles;
	} else if (inNumParticles < numParticles) {
		bufferSubData = true;
	}

	if (inNumActiveParticles > numParticles) {
		inNumActiveParticles = numParticles;
	}
	numActiveParticles = inNumActiveParticles;

	int numToUpload = bufferSubData ? inNumActiveParticles : numParticles;


	/*
	glm::vec3 pPos;
	int pIdx;

	vector<glm::vec3> vertexPositions;
	vector<int> vertexProfiles;

	for (int i = 0; i < inNumParticles; i++) {
		infile >> pPos.x;
		infile >> pPos.y;
		infile >> pPos.z;
		infile >> pIdx;

		vertexPositions.push_back(pPos);
		vertexProfiles.push_back(pIdx);
	}

	if (bufferSubData) {
		glNamedBufferSubData(particleVerticesVBO, 0, sizeof(glm::vec3) * numToUpload, vertexPositions.data());
		glNamedBufferSubData(particleProfilesVBO, 0, sizeof(int) * numToUpload, vertexProfiles.data());
	} else {
		glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numToUpload, vertexPositions.data(), GL_STATIC_DRAW);
		glNamedBufferData(particleProfilesVBO, sizeof(int) * numToUpload, vertexProfiles.data(), GL_STATIC_DRAW);
	}
	*/

	
	glm::vec3 *vertexPositions = new glm::vec3[numToUpload];
	int *vertexProfiles = new int[numToUpload];

	infile.read((char *)&vertexPositions[0], numToUpload * sizeof(glm::vec3));
	infile.read((char *)&vertexProfiles[0], numToUpload * sizeof(int));

	if (bufferSubData) {
		glNamedBufferSubData(particleVerticesVBO, 0, sizeof(glm::vec3) * numToUpload, vertexPositions);
		glNamedBufferSubData(particleProfilesVBO, 0, sizeof(int) * numToUpload, vertexProfiles);
	} else {
		glNamedBufferData(particleVerticesVBO, sizeof(glm::vec3) * numToUpload, vertexPositions, GL_STATIC_DRAW);
		glNamedBufferData(particleProfilesVBO, sizeof(int) * numToUpload, vertexProfiles, GL_STATIC_DRAW);
	}
	delete[] vertexPositions;
	delete[] vertexProfiles;
	


	


}

void ParticleSystem::loadParticleSaveFiles() {
	particleSaveFiles.clear();
	string path = PARTICLE_DATA_DIR;
	string ext = "";
	for (const auto &entry : fs::directory_iterator(path)) {
		if (getFileExtension(entry.path().string(), ext)) {
			if (ext == "bin") {
				particleSaveFiles.push_back(entry.path().string());
			}
		}
	}
	//cout << "Possible Particle Save Files:" << endl;
	//for (int i = 0; i < particleSaveFiles.size(); i++) {
	//	cout << " | " << particleSaveFiles[i] << endl;
	//}

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