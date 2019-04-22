#pragma once

//#include "Emitter.h"
#include "PositionalEmitter.h"

//#include <random>

class ParticleSystem;

class CircleEmitter : public PositionalEmitter {
public:

	float radius = 10000.0f;

	int numVisPoints = 120;

	CircleEmitter();
	CircleEmitter(std::string name, ParticleSystem *owner, glm::vec3 position = glm::vec3(0.0f), float radius = 1000.0f);
	CircleEmitter(const CircleEmitter &e, ParticleSystem *owner);
	~CircleEmitter();

	virtual void init();

	virtual void emitParticle();

	virtual void update();
	virtual void draw(ShaderProgram *shader);


	virtual void changeScale(float scaleChange);

	virtual void initBuffers();


	virtual void constructEmitterPropertiesTab(struct nk_context *ctx, UserInterface *ui);

protected:


	float prevRadius;


	virtual void updateVBOPoints();



};

