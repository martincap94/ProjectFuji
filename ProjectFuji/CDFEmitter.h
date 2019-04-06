#pragma once

#include "Emitter.h"

#include "CDFSampler.h"

#include <string>

class CDFEmitter : public Emitter {
public:


	CDFEmitter(ParticleSystem *owner, std::string probabilityTexturePath);
	~CDFEmitter();


	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();
	
protected:

	CDFSampler *sampler = nullptr;

};

