#include "CircleEmitter.h"

#include "Utils.h"
#include "ShaderManager.h"

//#include <random>
#include <vector>
#include <glad\glad.h>

using namespace std;

CircleEmitter::CircleEmitter(ParticleSystem * owner, glm::vec3 position, float radius, bool projectOntoTerrain) : PositionalEmitter(owner, position), radius(radius), projectOntoTerrain(projectOntoTerrain) {

	if (projectOntoTerrain && !heightMap) {
		cerr << "Project onto terrain is set to true but no heightMap was set! Project onto terrain will be turned off." << endl;
		projectOntoTerrain = false;
	}

	initBuffers();
	prevRadius = radius;

	shader = ShaderManager::getShaderPtr("singleColor");
}


CircleEmitter::~CircleEmitter() {
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
	if (enabled) {
		if (wiggle) {
			wigglePosition();
		}
	}


	if (prevPosition != position || prevRadius != radius) {
		prevPosition = position;
		prevRadius = radius;

		if (visible) {
			updateVBOPoints();
		}
	}

}

void CircleEmitter::draw() {
	if (!shader) {
		return;
	}
	draw(shader);
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

void CircleEmitter::initBuffers() {

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glBindVertexArray(0);

	updateVBOPoints();



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


