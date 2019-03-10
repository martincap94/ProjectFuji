#include "Emitter.h"



Emitter::Emitter(ParticleSystem *owner) : owner(owner) {
}


Emitter::Emitter(ParticleSystem * owner, HeightMap * heightMap, glm::vec3 position) : owner(owner), heightMap(heightMap), position(position) {
}

Emitter::~Emitter() {
}

//void Emitter::draw() {
//}
//
//void Emitter::draw(ShaderProgram * shader) {
//}
//
//void Emitter::initBuffers() {
//}


