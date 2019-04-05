#pragma once

#include "Emitter.h"

#include <string>

class CDFEmitterCUDA : public Emitter {
public:

	float *sums = nullptr;
	float maxTotalSum = 0;

	int width;
	int ewidth; // extended width
	int height;
	int numChannels;

	std::uniform_real_distribution<float> firstdist;

	CDFEmitterCUDA(ParticleSystem *owner, std::string probabilityTexturePath);
	~CDFEmitterCUDA();


	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();
};

