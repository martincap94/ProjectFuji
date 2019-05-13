#include "Emitter.h"

#include "ParticleSystem.h"


Emitter::Emitter() {}

Emitter::Emitter(string name, ParticleSystem *owner) : name(name), owner(owner) {
	init();
}

Emitter::Emitter(const Emitter & e, ParticleSystem * owner) : owner(owner) {
	name = e.name;
	minProfileIndex = e.minProfileIndex;
	maxProfileIndex = e.maxProfileIndex;
	maxParticlesToEmit = e.maxParticlesToEmit;
	numParticlesToEmitPerStep = e.numParticlesToEmitPerStep;
	init();
}

//Emitter::Emitter(const Emitter & e) {
//	owner = e.owner;
//
//	name = e.name;
//	minProfileIndex = e.minProfileIndex;
//	maxProfileIndex = e.maxProfileIndex;
//	maxParticlesToEmit = e.maxParticlesToEmit;
//	numParticlesToEmitPerStep = e.numParticlesToEmitPerStep;
//
//	mt = e.mt;
//	dist = e.dist;
//	distRange = e.distRange;
//	heightMap = e.heightMap;
//	maxProfileIndex = e.maxProfileIndex;
//	prevMinProfileIndex = e.prevMinProfileIndex;
//	prevMaxProfileIndex = e.prevMaxProfileIndex;
//	profileDist = e.profileDist;
//
//	initialized = e.initialized;
//
//}

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

void Emitter::setProfileIndexPos(int changeAmount) {

	minProfileIndex += changeAmount;
	maxProfileIndex += changeAmount;

	minProfileIndex = glm::clamp(minProfileIndex, 0, owner->stlpSim->stlpDiagram->numProfiles - 1);
	maxProfileIndex = glm::clamp(maxProfileIndex, 0, owner->stlpSim->stlpDiagram->numProfiles - 1);

	updateProfileIndexDistribution();
}

void Emitter::setProfileIndexRange(int changeAmount) {
	if (minProfileIndex == maxProfileIndex && changeAmount < 0) {
		return;
	}
	minProfileIndex -= changeAmount;
	maxProfileIndex += changeAmount;
	if (maxProfileIndex < minProfileIndex) {
		maxProfileIndex = minProfileIndex;
	}
	minProfileIndex = glm::clamp(minProfileIndex, 0, owner->stlpSim->stlpDiagram->numProfiles - 1);
	maxProfileIndex = glm::clamp(maxProfileIndex, 0, owner->stlpSim->stlpDiagram->numProfiles - 1);


	updateProfileIndexDistribution();
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

bool Emitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {

	nk_layout_row_dynamic(ctx, 15.0f, 1);
	if (initialized) {

		nk_checkbox_label(ctx, "Enabled", &enabled);
		nk_checkbox_label(ctx, "Visible", &visible);

		nk_property_int(ctx, "#Emit per Step", 0, &numParticlesToEmitPerStep, 10000, 10, 10);


		nk_property_int(ctx, "#Min Profile Index", 0, &minProfileIndex, maxProfileIndex, 1, 1);
		nk_property_int(ctx, "#Max Profile Index", minProfileIndex, &maxProfileIndex, owner->stlpSim->stlpDiagram->numProfiles - 1, 1, 1);
	}
	return true;
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





