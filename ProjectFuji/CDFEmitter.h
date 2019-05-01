///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       CDFEmitter.h
* \author     Martin Cap
* \brief      Describes emitter that uses CDF sampler.
*
*	This file describes the CDFEmitter class that uses CDFSampler class to generate particles in a
*	pattern given by a probability grayscale texture.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Emitter.h"

#include "CDFSampler.h"
#include "DynamicCDFSampler.h"

#include <string>

class CDFEmitter : public Emitter {
public:


	CDFEmitter();
	CDFEmitter(std::string name, ParticleSystem *owner, std::string probabilityTexturePath, bool useDynamicSampler = false);
	CDFEmitter(const CDFEmitter &e, ParticleSystem *owner);
	~CDFEmitter();

	virtual void init();

	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	
protected:

	std::string probabilityTexturePath = "";
	int useDynamicSampler = 0;

	CDFSampler *sampler = nullptr;
	DynamicCDFSampler *dsampler = nullptr; // only set when useDynamicSampler is true - helper so we do not have to cast in each loop

};

