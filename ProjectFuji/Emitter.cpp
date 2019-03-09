#include "Emitter.h"



Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
}

Emitter::Emitter(ParticleSystem * owner, HeightMap * heightMap) : owner(owner), heightMap(heightMap) {
}

Emitter::~Emitter() {
}
