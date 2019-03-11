#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
}


Emitter::Emitter(ParticleSystem * owner, glm::vec3 position) : owner(owner), position(position) {
	mt = mt19937(rd());
	dist = uniform_real_distribution<float>(0.0f, 1.0f);
	distRange = uniform_real_distribution<float>(-1.0f, 1.0f);
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

void Emitter::wigglePosition() {

	position.x += distRange(mt) * xWiggleRange;
	position.z += distRange(mt) * zWiggleRange;

}

//void Emitter::draw() {
//}
//
//void Emitter::draw(ShaderProgram * shader) {
//}
//
//void Emitter::initBuffers() {
//}


