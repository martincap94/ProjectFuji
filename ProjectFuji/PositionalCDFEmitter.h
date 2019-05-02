///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       PositionalCDFEmitter.h
* \author     Martin Cap
*
*	Describes the PositionalCDFEmitter class. It is a type of positional emitter that uses CDF
*	sampler to create patterns based on probability texture. It can also be used as a general
*	brush that can draw particles on terrain when in brush mode.
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "PositionalEmitter.h"
#include "CDFSampler.h"
#include "ShaderProgram.h"
#include "Texture.h"

#include "UserInterface.h"

#include <nuklear.h>

#include <string>

class ParticleSystem;

class PositionalCDFEmitter : public PositionalEmitter {
public:

	std::string probabilityTexturePath = "";
	const int numVisPoints = 4;


	float scale = 1.0f;
	int centered = 1;

	PositionalCDFEmitter();
	PositionalCDFEmitter(std::string name, ParticleSystem *owner, std::string probabilityTexturePath);
	PositionalCDFEmitter(const PositionalCDFEmitter &e, ParticleSystem *owner);
	~PositionalCDFEmitter();


	virtual void init();

	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();

	virtual void changeScale(float scaleChange);


	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	Texture *getSamplerTexture();

protected:

	float prevScale;

	CDFSampler *sampler = nullptr;

	struct nk_image nkSamplerTexture;

	virtual void updateVBOPoints();

};

