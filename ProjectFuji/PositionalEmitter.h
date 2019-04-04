#pragma once

#include "Emitter.h"

class PositionalEmitter : public Emitter{
public:

	glm::vec3 position;

	int wiggle = 0; // when enabled, position of the emitter "wiggles" around (shifts) after each emission iteration by the given offset range

	float xWiggleRange = 0.5f;
	float zWiggleRange = 0.5f;



	PositionalEmitter(ParticleSystem *owner);
	PositionalEmitter(ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f));
	~PositionalEmitter();

	virtual void wigglePosition();

protected:

	glm::vec3 prevPosition;



};


