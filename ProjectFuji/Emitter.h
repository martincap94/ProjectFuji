#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"

#include <random>

class ParticleSystem;

class Emitter {
public:

	ParticleSystem *owner;
	HeightMap *heightMap;

	ShaderProgram *shader;

	random_device rd;
	mt19937_64 mt;
	uniform_real_distribution<float> dist;
	uniform_real_distribution<float> distRange;

	int maxParticlesToEmit = 0; // 0 means unlimited
	int numParticlesToEmitPerStep = 1;

	// booleans (as integers for Nuklear)
	int enabled = 0;
	int visible = 0;

	//Emitter();
	Emitter(ParticleSystem *owner);
	~Emitter();

	__inline__ virtual bool canEmitParticle();
	virtual void emitParticle() = 0;
	virtual void emitParticles();
	virtual void emitParticles(int numParticles);

	virtual void update() = 0;
	virtual void draw() = 0;
	virtual void draw(ShaderProgram *shader) = 0;
	virtual void initBuffers() = 0;



protected:

	GLuint VAO;
	GLuint VBO;

};

