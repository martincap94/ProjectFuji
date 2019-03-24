#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"

#include <random>

class ParticleSystem;

class Emitter {
public:

	glm::vec3 position;

	ParticleSystem *owner;
	HeightMap *heightMap;

	ShaderProgram *shader;

	random_device rd;
	mt19937 mt;
	uniform_real_distribution<float> dist;
	uniform_real_distribution<float> distRange;

	int maxParticlesToEmit = 0; // 0 means unlimited
	int numParticlesToEmitPerStep = 1;

	// booleans (as integers for Nuklear)
	int enabled = 0;
	int visible = 0;
	int wiggle = 0; // when enabled, position of the emitter "wiggles" around (shifts) after each emission iteration by the given offset range

	float xWiggleRange = 0.5f;
	float zWiggleRange = 0.5f;

	//Emitter();
	Emitter(ParticleSystem *owner);
	Emitter(ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f));
	~Emitter();

	virtual void emitParticle() = 0;
	virtual void emitParticles();
	virtual void emitParticles(int numParticles);

	virtual void update() = 0;
	virtual void wigglePosition();
	virtual void draw() = 0;
	virtual void draw(ShaderProgram *shader) = 0;
	virtual void initBuffers() = 0;



protected:

	GLuint VAO;
	GLuint VBO;

	glm::vec3 prevPosition;

};

