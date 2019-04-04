#include "PositionalEmitter.h"

#include "ParticleSystem.h"


PositionalEmitter::PositionalEmitter(ParticleSystem * owner, glm::vec3 position) : Emitter(owner), position(position) {
	prevPosition = position;
}

PositionalEmitter::~PositionalEmitter() {
}


void PositionalEmitter::wigglePosition() {

	position.x += distRange(mt) * xWiggleRange;
	position.z += distRange(mt) * zWiggleRange;

}