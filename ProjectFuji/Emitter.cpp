#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
}


Emitter::Emitter(ParticleSystem * owner, glm::vec3 position) : owner(owner), position(position) {
	heightMap = owner->heightMap;
	prevPosition = position;
}

Emitter::~Emitter() {
}

void Emitter::emitParticles() {
	for (int i = 0; i < numParticlesToEmitPerStep; i++) {
		emitParticle();
	}
}

void Emitter::emitParticles(int numParticles) {
	for (int i = 0; i < numParticles; i++) {
		emitParticle();
	}
}

//void Emitter::draw() {
//}
//
//void Emitter::draw(ShaderProgram * shader) {
//}
//
//void Emitter::initBuffers() {
//}


