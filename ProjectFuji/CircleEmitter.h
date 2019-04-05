#pragma once

//#include "Emitter.h"
#include "PositionalEmitter.h"
#include "ParticleSystem.h"

//#include <random>

class CircleEmitter : public PositionalEmitter {
public:

	bool projectOntoTerrain;
	float radius;

	int numVisPoints = 120;


	CircleEmitter(ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f), float radius = 1000.0f, bool projectOntoTerrain = true);
	~CircleEmitter();

	virtual void emitParticle();

	virtual void update();
	virtual void draw();
	virtual void draw(ShaderProgram *shader);

	virtual void initBuffers();

protected:

	float prevRadius;


	void updateVBOPoints();



};

