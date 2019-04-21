#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"
#include "UserInterface.h"

#include <nuklear.h>

#include <random>

class ParticleSystem;

class Emitter {
public:

	enum eEmitterType {
		CIRCULAR = 0,
		CDF_TERRAIN,
		CDF_POSITIONAL,

		_NUM_EMITTER_TYPES
	};

	ParticleSystem *owner;
	HeightMap *heightMap;

	ShaderProgram *shader;

	random_device rd;
	mt19937_64 mt;
	uniform_real_distribution<float> dist;
	uniform_real_distribution<float> distRange;
	uniform_int_distribution<int> profileDist;

	int minProfileIndex = 0;
	int maxProfileIndex = 0;

	int maxParticlesToEmit = 0; // 0 means unlimited
	int numParticlesToEmitPerStep = 1;

	// booleans (as integers for Nuklear)
	int initialized = 0;
	int enabled = 0;
	int visible = 0;

	Emitter();
	Emitter(ParticleSystem *owner);
	Emitter(const Emitter &e, ParticleSystem *owner);

	~Emitter();

	
	virtual void init();

	__inline__ virtual bool canEmitParticle();
	virtual void emitParticle() = 0;
	virtual void emitParticles();
	virtual void emitParticles(int numParticles);

	virtual void update() = 0;
	virtual void draw() = 0;
	virtual void draw(ShaderProgram *shader) = 0;
	virtual void initBuffers() = 0;

	inline virtual int getRandomProfileIndex();

	static const char *getEmitterTypeString(int emitterType);

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui) = 0;

protected:

	// for refresh when UI changes some values
	int prevMinProfileIndex;
	int prevMaxProfileIndex;

	GLuint VAO;
	GLuint VBO;

	inline void updateProfileIndexDistribution();

};

