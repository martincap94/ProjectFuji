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

//#define THRUST_BIN_SEARCH //!< Whether to use binary search on GPU using Thrust
							//!< Warning! This is much slower than regular CPU version


CDFEmitter::CDFEmitter() : Emitter() {}

// expects path to 16-bit grayscale png
CDFEmitter::CDFEmitter(string name, ParticleSystem *owner, string probabilityTexturePath, bool useDynamicSampler) : Emitter(name, owner), probabilityTexturePath(probabilityTexturePath) {
	//sampler = new CDFSampler(this->probabilityTexturePath);
	this->useDynamicSampler = (int)useDynamicSampler;
	init();
}

CDFEmitter::CDFEmitter(const CDFEmitter &e, ParticleSystem *owner) : Emitter(e, owner) {
	probabilityTexturePath = e.probabilityTexturePath;
	useDynamicSampler = e.useDynamicSampler;
	init();
}




void CDFEmitter::init() {
	if (useDynamicSampler) {
		dsampler = new DynamicCDFSampler(probabilityTexturePath);
		sampler = (CDFSampler *)dsampler;
	} else {
		sampler = new CDFSampler(probabilityTexturePath);
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

bool CDFEmitter::constructEmitterPropertiesTab(nk_context *ctx, UserInterface *ui) {
	bool canBeConstructed = Emitter::constructEmitterPropertiesTab(ctx, ui);

	static Texture *selectedTexture = nullptr;
	if (!initialized) {
		nk_checkbox_label(ctx, "Dynamic Sampler", &useDynamicSampler);
		ui->constructTextureSelection_label(&selectedTexture, "Probability Texture: ", 0.3f, probabilityTexturePath, true);
		if (selectedTexture != nullptr) {
			probabilityTexturePath = selectedTexture->filename;
		}
		canBeConstructed = canBeConstructed && (selectedTexture != nullptr);

		if (!canBeConstructed) {
			nk_layout_row_dynamic(ctx, 15.0f, 1);
			nk_label_colored(ctx, "Please select a probability texture.", NK_TEXT_LEFT, nk_rgb(255, 150, 150));
		}
	} else {
		if (useDynamicSampler) {


			dsampler->pSampler.constructUIPropertiesTab(ctx, true);
			nk_property_float(ctx, "Decrease Perlin Prob.", 0.0f, &dsampler->perlinProbabilityDecrease, 1.0f, 0.01f, 0.01f);


			nk_checkbox_label(ctx, "Use Time as Seed", &dsampler->useTimeAsSeed);
			if (!dsampler->useTimeAsSeed) {
				nk_property_int(ctx, "Seed", 0, &dsampler->seed, 10000, 1, 1);
			}

			if (nk_button_label(ctx, "Generate New Noise Func. * CDF Texture")) {
				dsampler->updatePerlinNoiseCPU(false);
			}
			if (nk_button_label(ctx, "Generate New Noise Func.")) {
				dsampler->updatePerlinNoiseCPU(true);
			}
		}
	}

	return canBeConstructed;
}

