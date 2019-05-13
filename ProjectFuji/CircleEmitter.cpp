#include "CircleEmitter.h"

#include "Utils.h"
#include "ShaderManager.h"
#include "ParticleSystem.h"

//#include <random>
#include <vector>
#include <glad\glad.h>

#include <nuklear.h>

using namespace std;

CircleEmitter::CircleEmitter() : PositionalEmitter() {
}

CircleEmitter::CircleEmitter(string name, ParticleSystem * owner, glm::vec3 position, float radius) : PositionalEmitter(name, owner, position), radius(radius) {
	init();
}

CircleEmitter::CircleEmitter(const CircleEmitter & e, ParticleSystem *owner) : PositionalEmitter(e, owner) {
	radius = e.radius;
	numVisPoints = e.numVisPoints;
	init();
}


CircleEmitter::~CircleEmitter() {
}

void CircleEmitter::init() {

	initBuffers();
	prevRadius = radius;

	shader = ShaderManager::getShaderPtr("singleColor");

}

void CircleEmitter::emitParticle() {

	if (!canEmitParticle()) {
		return;
	}


	float randx;
	float randz;
	
	float a = dist(mt) * 2.0f * (float)PI;
	float r = radius * sqrtf(dist(mt));

	
	randx = r * cos(a);
	randz = r * sin(a);
	
	randx += position.x;
	randz += position.z;
	
	float y = heightMap->getHeight(randx, randz);
		
	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);
	
	p.profileIndex = getRandomProfileIndex();

	owner->pushParticleToEmit(p);
}


void CircleEmitter::update() {
	PositionalEmitter::update();


	if (prevPosition != position || prevRadius != radius) {
		prevPosition = position;
		prevRadius = radius;

		if (visible) {
			updateVBOPoints();
		}
	}

}


void CircleEmitter::draw() {
	Emitter::draw();
}

void CircleEmitter::draw(ShaderProgram * shader) {
	if (!visible) {
		return;
	}
	shader->use();
	shader->setColor(glm::vec3(0.0f, 1.0f, 1.0f));
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINE_LOOP, 0, numVisPoints);

}

void CircleEmitter::changeScale(float scaleChange) {
	radius += scaleChange * sqrtf(radius) * 2.0f;
	/*if (scaleChange < 0.0f) {
		radius += glm::clamp(scaleChange * radius, -0.01f, -100.0f);
	} else {
		radius += glm::clamp(scaleChange * radius, 0.01f, 100.0f);
	}*/
	radius = glm::clamp(radius, 0.1f, 100000.0f);

}

void CircleEmitter::initBuffers() {
	Emitter::initBuffers();

	updateVBOPoints();
}

bool CircleEmitter::constructEmitterPropertiesTab(nk_context * ctx, UserInterface * ui) {
	bool canBeConstructed = PositionalEmitter::constructEmitterPropertiesTab(ctx, ui);
	nk_property_float(ctx, "Radius", 1.0f, &radius, 100000.0f, 1.0f, 1.0f);

	return canBeConstructed;
}

void CircleEmitter::updateVBOPoints() {
	vector<glm::vec3> vertices;

	//numVisPoints = radius;

	float deltaTheta = 360.0f / (float)numVisPoints;
	float theta = 0.0f;
	for (int i = 0; i < numVisPoints; i++) {
		float x = radius * cos(glm::radians(theta)) + position.x;
		float z = radius * sin(glm::radians(theta)) + position.z;
		float y = heightMap->getHeight(x, z);

		vertices.push_back(glm::vec3(x, y, z));
		theta += deltaTheta;
	}

	glNamedBufferData(VBO, sizeof(glm::vec3) * numVisPoints, vertices.data(), GL_STATIC_DRAW);

}


