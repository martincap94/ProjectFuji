#pragma once

#include "Emitter.h"

#include "CDFSampler.h"

#include <string>

class CDFEmitter : public Emitter {
public:

	std::string probabilityTexturePath = "";

	CDFEmitter();
	CDFEmitter(std::string name, ParticleSystem *owner, std::string probabilityTexturePath);
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

	CDFSampler *sampler = nullptr;

};

