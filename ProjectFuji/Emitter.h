///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       Emitter.h
* \author     Martin Cap
* \brief      Declaration of the Emitter class.
*
*	This file declares the abstract Emitter class that is used to generate particles.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"
#include "UserInterface.h"

#include <nuklear.h>

#include <random>
#include <string>

class ParticleSystem;

class Emitter {
public:

	enum eEmitterType {
		CIRCULAR = 0,
		CDF_TERRAIN,
		CDF_POSITIONAL,

		_NUM_EMITTER_TYPES
	};

	std::string name;

	ParticleSystem *owner;
	HeightMap *heightMap;

	ShaderProgram *shader;

	std::random_device rd;
	std::mt19937_64 mt;
	std::uniform_real_distribution<float> dist;
	std::uniform_real_distribution<float> distRange;
	std::uniform_int_distribution<int> profileDist;

	int minProfileIndex = 0;
	int maxProfileIndex = 0;

	int maxParticlesToEmit = 0; // 0 means unlimited
	int numParticlesToEmitPerStep = 1;

	// booleans (as integers for Nuklear)
	int initialized = 0;
	int enabled = 0;
	int visible = 0;

	Emitter();
	Emitter(std::string name, ParticleSystem *owner);
	
	/*
		Copy before initialization of the copied emitter (initialized here)
		used for creating emitter from dummy emitters that are used in UI only.
	*/
	Emitter(const Emitter &e, ParticleSystem *owner);

	// Pure copy
	//Emitter(const Emitter &e);

	~Emitter();

	
	virtual void init();

	__inline__ virtual bool canEmitParticle();
	virtual void emitParticle() = 0;
	virtual void emitParticles();
	virtual void emitParticles(int numParticles);

	virtual void update() = 0;
	virtual void draw();
	virtual void draw(ShaderProgram *shader) = 0;
	virtual void initBuffers();

	void setProfileIndexPos(int changeAmount);
	void setProfileIndexRange(int changeAmount);

	inline virtual int getRandomProfileIndex();

	static const char *getEmitterTypeString(int emitterType);

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

	static const char *getEmitterName(Emitter *emitter);


protected:

	// for refresh when UI changes some values
	int prevMinProfileIndex;
	int prevMaxProfileIndex;

	GLuint VAO;
	GLuint VBO;

	inline void updateProfileIndexDistribution();

};


