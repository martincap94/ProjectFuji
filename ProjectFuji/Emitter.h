#pragma once

#include <glm\glm.hpp>

#include "HeightMap.h"

class ParticleSystem;

class Emitter {
public:

	glm::vec3 position;

	ParticleSystem *owner;
	HeightMap *heightMap;

	//Emitter();
	Emitter(ParticleSystem *owner);
	Emitter(ParticleSystem *owner, HeightMap *heightMap);
	~Emitter();

	virtual void doStep() = 0;
};

