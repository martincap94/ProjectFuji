#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
	mt = mt19937_64(rd());
	dist = uniform_real_distribution<float>(0.0f, 1.0f);
	distRange = uniform_real_distribution<float>(-1.0f, 1.0f);
	heightMap = owner->heightMap;
	maxProfileIndex = owner->stlpSim->stlpDiagram->numProfiles - 1;
	prevMinProfileIndex = minProfileIndex;
	prevMaxProfileIndex = maxProfileIndex;
	profileDist = uniform_int_distribution<int>(minProfileIndex, maxProfileIndex);

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
	updateProfileIndexDistribution();
	for (int i = 0; i < numParticles; i++) {
		emitParticle();
	}
}

inline int Emitter::getRandomProfileIndex() {
	return profileDist(mt);
}

inline void Emitter::updateProfileIndexDistribution() {
	if (prevMinProfileIndex != minProfileIndex || prevMaxProfileIndex != maxProfileIndex) {
		profileDist = uniform_int_distribution<int>(minProfileIndex, maxProfileIndex);
		prevMinProfileIndex = minProfileIndex;
		prevMaxProfileIndex = maxProfileIndex;
	}

}




