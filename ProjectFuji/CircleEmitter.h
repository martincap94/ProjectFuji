#pragma once

#include "Emitter.h"
#include "ParticleSystem.h"

class CircleEmitter : public Emitter {
public:

	bool projectOntoTerrain;
	float radius;

	int numVisPoints = 36;

	CircleEmitter(ParticleSystem *owner, HeightMap *heightMap, glm::vec3 position = glm::vec3(0.0f), float radius = 1.0f, bool projectOntoTerrain = false);
	~CircleEmitter();

	virtual void emitParticle();
	virtual void emitParticles(int numParticles);

	virtual void draw();
	virtual void draw(ShaderProgram *shader);

	virtual void initBuffers();

};

