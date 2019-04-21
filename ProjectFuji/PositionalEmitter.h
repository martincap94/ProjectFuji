#pragma once

#include "Emitter.h"

class PositionalEmitter : public Emitter {
public:

	glm::vec3 position = glm::vec3(0.0f);

	int wiggle = 0; // when enabled, position of the emitter "wiggles" around (shifts) after each emission iteration by the given offset range

	float xWiggleRange = 0.5f;
	float zWiggleRange = 0.5f;

	PositionalEmitter();
	PositionalEmitter(std::string name, ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f));
	PositionalEmitter(const PositionalEmitter &e, ParticleSystem *owner);
	~PositionalEmitter();

	virtual void init();
	virtual void update();

	virtual void wigglePosition();

	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);


protected:

	glm::vec3 prevPosition;



};


