#include "CDFEmitter.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>


#include "CUDAUtils.cuh"
#include "ParticleSystem.h"
#include "TextureManager.h"
#include "Utils.h"

#include <stb_image.h>

using namespace std;

//#define THRUST_BIN_SEARCH // much slower than regular CPU version


// expects path to 16-bit grayscale png
CDFEmitter::CDFEmitter(ParticleSystem *owner, string probabilityTexturePath) : Emitter(owner) {
	sampler = new CDFSampler(probabilityTexturePath);
}


CDFEmitter::~CDFEmitter() {
	if (sampler) {
		delete sampler;
	}
}

void CDFEmitter::emitParticle() {

	if (!canEmitParticle()) {
		return;
	}

	glm::ivec2 sample = sampler->getSample();


	Particle p;
	glm::vec3 pos;

	p.profileIndex = rand() % (owner->stlpSim->stlpDiagram->numProfiles - 1);
	p.velocity = glm::vec3(0.0f);

	//cout << pos.x << ", " << pos.y << ", " << pos.z << endl;

	pos = glm::vec3(sample.x, 0.0f, sample.y);

	// move inside the texel
	pos.x += getRandFloat(0.0f, 1.0f);
	pos.z += getRandFloat(0.0f, 1.0f);

	pos.x *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
	pos.z *= owner->heightMap->vars->texelWorldSize; // ugly, cleanup
	pos.y = owner->heightMap->getHeight(pos.x, pos.z, true);
	p.position = pos;


	owner->pushParticleToEmit(p);

}

void CDFEmitter::update() {
}

void CDFEmitter::draw() {
}

void CDFEmitter::draw(ShaderProgram * shader) {
}

void CDFEmitter::initBuffers() {
}

