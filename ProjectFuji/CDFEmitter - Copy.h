#pragma once

#include "Emitter.h"

#include <string>

class CDFEmitter : public Emitter {
public:

	float *arr = nullptr;
	float *sums = nullptr;
	float maxTotalSum = 0;

	int width;
	int height;
	int numChannels;

	std::uniform_real_distribution<float> firstdist;

	CDFEmitter(ParticleSystem *owner, std::string probabilityTexturePath);
	~CDFEmitter();


	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);
	virtual void initBuffers();
	
protected:
	
	float *d_arr; 
	// since we want to change the original array, we cannot do the prefix sum (scan) in-place
	float *d_sums;


	virtual void initCUDA();

};

