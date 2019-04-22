#pragma once

#include "PositionalEmitter.h"
#include "CDFSampler.h"
#include "ShaderProgram.h"

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

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

protected:

	CDFSampler *sampler = nullptr;

	virtual void updateVBOPoints();

};

