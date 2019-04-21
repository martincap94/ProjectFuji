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

#include <nuklear.h>

using namespace std;

//#define THRUST_BIN_SEARCH // much slower than regular CPU version


CDFEmitter::CDFEmitter() : Emitter() {}

CDFEmitter::CDFEmitter(const CDFEmitter & e, ParticleSystem * owner) : Emitter(e, owner) {
	probabilityTexturePath = e.probabilityTexturePath;
	init();
}

// expects path to 16-bit grayscale png
CDFEmitter::CDFEmitter(ParticleSystem *owner, string probabilityTexturePath) : Emitter(owner), probabilityTexturePath(probabilityTexturePath) {
	//sampler = new CDFSampler(this->probabilityTexturePath);
	init();
}



void CDFEmitter::init() {
	sampler = new CDFSampler(probabilityTexturePath);
}

void CDFEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	Texture *selectedTexture = nullptr;
	ui->constructTextureSelection(&selectedTexture, probabilityTexturePath);
	if (selectedTexture != nullptr) {
		probabilityTexturePath = selectedTexture->filename;
	}
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

	Particle p;
	p.position = heightMap->getWorldPositionSample(sampler);
	//p.profileIndex = rand() % (owner->stlpSim->stlpDiagram->numProfiles - 1);
	p.profileIndex = getRandomProfileIndex();
	p.velocity = glm::vec3(0.0f);

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
