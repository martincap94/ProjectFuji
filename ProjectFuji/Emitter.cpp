#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
	mt = mt19937_64(rd());
	dist = uniform_real_distribution<float>(0.0f, 1.0f);
	distRange = uniform_real_distribution<float>(-1.0f, 1.0f);
	heightMap = owner->heightMap;
}




Emitter::~Emitter() {
}

bool Emitter::canEmitParticle() {
	return (owner->numActiveParticles < owner->numParticles);
}


void Emitter::emitParticles() {
	emitParticles(numParticlesToEmitPerStep);
}

void Emitter::emitParticles(int numParticles) {
	if (!enabled || !owner) {
		return;
	}
	if (owner->numActiveParticles >= owner->numParticles) {
		return;
	}
	for (int i = 0; i < numParticles; i++) {
		emitParticle();
	}
}

//void Emitter::wigglePosition() {
//
//	position.x += distRange(mt) * xWiggleRange;
//	position.z += distRange(mt) * zWiggleRange;
//
//}

//void Emitter::draw() {
//}
//
//void Emitter::draw(ShaderProgram * shader) {
//}
//
//void Emitter::initBuffers() {
//}


