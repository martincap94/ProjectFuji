#pragma once

#include "Emitter.h"

#include <string>

class CDFEmitter : public Emitter {
public:

	unsigned long long int *sums = nullptr;
	unsigned long long int maxTotalSum = 0;

	int width;
	int ewidth; // extended width
	int height;
	int numChannels;
	
	std::uniform_int_distribution<unsigned long long int> firstdist;
	std::uniform_int_distribution<unsigned long long int> secondDist;

	CDFEmitter(ParticleSystem *owner, std::string probabilityTexturePath);
	~CDFEmitter();


	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();
};

