#pragma once

#include <glad\glad.h>
#include <glm\glm.hpp>

#include "HeightMap.h"
#include "ShaderProgram.h"

class ParticleSystem;

class Emitter {
public:

	glm::vec3 position;

	ParticleSystem *owner;
	HeightMap *heightMap;

	ShaderProgram *shader;

	int maxParticlesToEmit = 0; // 0 means unlimited

	//Emitter();
	Emitter(ParticleSystem *owner);
	Emitter(ParticleSystem *owner, HeightMap *heightMap, glm::vec3 position = glm::vec3(0.0f));
	~Emitter();

	virtual void emitParticle() = 0;
	virtual void emitParticles(int numParticles) = 0;

	virtual void draw() = 0;
	virtual void draw(ShaderProgram *shader) = 0;
	virtual void initBuffers() = 0;


protected:

	GLuint VAO;
	GLuint VBO;


};

