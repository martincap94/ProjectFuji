#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter() {}

Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
	init();
}

Emitter::Emitter(const Emitter & e, ParticleSystem * owner) : owner(owner) {
	minProfileIndex = e.minProfileIndex;
	maxProfileIndex = e.maxProfileIndex;
	maxParticlesToEmit = e.maxParticlesToEmit;
	numParticlesToEmitPerStep = e.numParticlesToEmitPerStep;
	init();
}

Emitter::~Emitter() {
}

void Emitter::init() {
	initialized = 1;

	mt = mt19937_64(rd());
	dist = uniform_real_distribution<float>(0.0f, 1.0f);
	distRange = uniform_real_distribution<float>(-1.0f, 1.0f);
	heightMap = owner->heightMap;
	maxProfileIndex = owner->stlpSim->stlpDiagram->numProfiles - 1;
	prevMinProfileIndex = minProfileIndex;
	prevMaxProfileIndex = maxProfileIndex;
	profileDist = uniform_int_distribution<int>(minProfileIndex, maxProfileIndex);

}

bool Emitter::canEmitParticle() {
	return (initialized && owner->numActiveParticles < owner->numParticles);
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

void Emitter::draw() {
	if (!shader) {
		return;
	}
	draw(shader);
}

void Emitter::initBuffers() {
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glBindVertexArray(0);
}

inline int Emitter::getRandomProfileIndex() {
	return profileDist(mt);
}

const char * Emitter::getEmitterTypeString(int emitterType) {
	switch (emitterType) {
		case eEmitterType::CIRCULAR:
			return "Circular";
		case eEmitterType::CDF_TERRAIN:
			return "CDF (Terrain)";
		case eEmitterType::CDF_POSITIONAL:
			return "CDF (Positional)";
	}
	return "None";
}

void Emitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {

}

const char * Emitter::getEmitterName(Emitter *emitter) {
	if (emitter == nullptr) {
		return "None";
	} else {
		return emitter->name.empty() ? "Unnamed" : emitter->name.c_str();
	}
}

inline void Emitter::updateProfileIndexDistribution() {
	if (prevMinProfileIndex != minProfileIndex || prevMaxProfileIndex != maxProfileIndex) {
		profileDist = uniform_int_distribution<int>(minProfileIndex, maxProfileIndex);
		prevMinProfileIndex = minProfileIndex;
		prevMaxProfileIndex = maxProfileIndex;
	}

}





