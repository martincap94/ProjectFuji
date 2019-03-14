#include "CircleEmitter.h"

#include "Utils.h"
#include "ShaderManager.h"

//#include <random>
#include <vector>
#include <glad\glad.h>

using namespace std;

CircleEmitter::CircleEmitter(ParticleSystem * owner, glm::vec3 position, float radius, bool projectOntoTerrain) : Emitter(owner, position), radius(radius), projectOntoTerrain(projectOntoTerrain) {

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
	if (!enabled) {
		return;
	}
	if (!owner) {
		cerr << "Emitter has no owning ParticleSystem! No emission will be done!" << endl;
		return;
	}
	if (owner->numActiveParticles >= owner->numParticles) {
		//cout << "Max active particles reached." << endl;
		return;
	}


	// testing generation in circle
	float randx;
	float randz;

	//static random_device rd;
	//static mt19937 mt(rd());
	//static uniform_real_distribution<float> dist(0.0f, 1.0f);
	
	float a = dist(mt) * 2.0f * (float)PI;
	float r = radius * sqrtf(dist(mt));
	
	randx = r * cos(a);
	randz = r * sin(a);
	
	randx += position.x;
	randz += position.z;
	
	
	// interpolate
	int leftx = (int)randx;
	int rightx = leftx + 1;
	int leftz = (int)randz;
	int rightz = leftz + 1;

	// clamp values
	leftx = glm::clamp(leftx, 0, heightMap->width - 1);
	rightx = glm::clamp(rightx, 0, heightMap->width - 1);
	leftz = glm::clamp(leftz, 0, heightMap->height - 1);
	rightz = glm::clamp(rightz, 0, heightMap->height - 1);
	

	float xRatio = randx - leftx;
	float zRatio = randz - leftz;
	
	float y1 = heightMap->data[leftx][leftz];
	float y2 = heightMap->data[leftx][rightz];
	float y3 = heightMap->data[rightx][leftz];
	float y4 = heightMap->data[rightx][rightz];
	
	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;
	
	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;
	



	//particlePositions.push_back(glm::vec3(randx, y, randz));
	owner->particleVerticesToEmit.push_back(glm::vec3(randx, y, randz));

	
	owner->stlpSim->mapFromSimulationBox(y);
	
	Particle p;
	p.position = glm::vec3(randx, y, randz);
	p.velocity = glm::vec3(0.0f);
	
	ppmImage *profileMap = owner->stlpSim->profileMap;
	STLPDiagram *stlpDiagram = owner->stlpSim->stlpDiagram;
	if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {
	
		glm::vec2 p1 = profileMap->data[leftx][leftz];
		glm::vec2 p2 = profileMap->data[leftx][rightz];
		glm::vec2 p3 = profileMap->data[rightx][leftz];
		glm::vec2 p4 = profileMap->data[rightx][rightz];
	
		glm::vec2 pi1 = zRatio * p2 + (1.0f - zRatio) * p1;
		glm::vec2 pi2 = zRatio * p4 + (1.0f - zRatio) * p3;
	
		glm::vec2 pif = xRatio * pi2 + (1.0f - xRatio) * pi1;
		glm::ivec2 pii = (glm::ivec2)pif;
	
		if (pii.y != pii.x) {
			p.profileIndex = (rand() % (pii.y - pii.x) + pii.x) % (stlpDiagram->numProfiles - 1);
		} else {
			p.profileIndex = pii.x % (stlpDiagram->numProfiles - 1);
		}
	
	} else {
		p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
	}
	
	p.updatePressureVal();
	
	owner->particleProfilesToEmit.push_back(p.profileIndex);
	owner->verticalVelocitiesToEmit.push_back(0.0f);
	owner->numActiveParticles++;

	//particles.push_back(p);
	//numParticles++;




}

//void CircleEmitter::emitParticles() {
//}
//
//void CircleEmitter::emitParticles(int numParticles) {
//	for (int i = 0; i < numParticles; i++) {
//		emitParticle();
//	}
//}

void CircleEmitter::update() {
	//if (!enabled) {
	//	return;
	//}
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
	if (!visible) {
		return;
	}
	if (!shader) {
		return;
	}

	//glDepthFunc(GL_ALWAYS);
	glPointSize(0.5f);
	shader->use();
	glBindVertexArray(VAO);
	//glDrawArrays(GL_POINTS, 0, numVisPoints);
	glDrawArrays(GL_LINE_LOOP, 0, numVisPoints);
	//glDepthFunc(GL_LEQUAL);
}

void CircleEmitter::draw(ShaderProgram * shader) {
	if (!visible) {
		return;
	}
	shader->use();
	glPointSize(0.5f);
	glBindVertexArray(VAO);
	//glDrawArrays(GL_POINTS, 0, numVisPoints);
	glDrawArrays(GL_LINE_LOOP, 0, numVisPoints);

}

void CircleEmitter::initBuffers() {

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	//glBufferData(GL_ARRAY_BUFFER, )

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glBindVertexArray(0);

	updateVBOPoints();



}

void CircleEmitter::updateVBOPoints() {
	vector<glm::vec3> vertices;

	numVisPoints = 10 * radius;

	float deltaTheta = 360.0f / (float)numVisPoints;
	float theta = 0.0f;
	for (int i = 0; i < numVisPoints; i++) {
		float x = radius * cos(glm::radians(theta)) + position.x;
		float z = radius * sin(glm::radians(theta)) + position.z;

		int leftx = (int)x;
		int rightx = leftx + 1;
		int leftz = (int)z;
		int rightz = leftz + 1;

		// clamp values
		leftx = glm::clamp(leftx, 0, heightMap->width - 1);
		rightx = glm::clamp(rightx, 0, heightMap->width - 1);
		leftz = glm::clamp(leftz, 0, heightMap->height - 1);
		rightz = glm::clamp(rightz, 0, heightMap->height - 1);


		float xRatio = x - leftx;
		float zRatio = z - leftz;

		float y1 = heightMap->data[leftx][leftz];
		float y2 = heightMap->data[leftx][rightz];
		float y3 = heightMap->data[rightx][leftz];
		float y4 = heightMap->data[rightx][rightz];

		float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
		float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;

		float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;

		vertices.push_back(glm::vec3(x, y, z));
		theta += deltaTheta;
	}

	glNamedBufferData(VBO, sizeof(glm::vec3) * numVisPoints, vertices.data(), GL_STATIC_DRAW);

}



//void ParticleSystem::generateParticleOnTerrain(std::vector<glm::vec3>& outVector) {
//
//
//	// testing generation in circle
//	float randx;
//	float randz;
//
//	bool incircle = false;
//	if (incircle) {
//
//		float R = 10.0f;
////		static random_device rd;
//		static mt19937 mt(rd());
//		static uniform_real_distribution<float> dist(0.0f, 1.0f);
//
//		float a = dist(mt) * 2.0f * (float)PI;
//		float r = R * sqrtf(dist(mt));
//
//		randx = r * cos(a);
//		randz = r * sin(a);
//
//		randx += heightMap->width / 2;
//		randz += heightMap->height / 2;
//
//	} else {
//		randx = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->width - 2.0f)));
//		randz = (float)(rand() / (float)(RAND_MAX / ((float)heightMap->height - 2.0f)));
//	}
//
//	// interpolate
//	int leftx = (int)randx;
//	int rightx = leftx + 1;
//	int leftz = (int)randz;
//	int rightz = leftz + 1;
//
//	// leftx and leftz cannot be < 0 and rightx and rightz cannot be >= GRID_WIDTH or GRID_DEPTH
//	float xRatio = randx - leftx;
//	float zRatio = randz - leftz;
//
//	float y1 = heightMap->data[leftx][leftz];
//	float y2 = heightMap->data[leftx][rightz];
//	float y3 = heightMap->data[rightx][leftz];
//	float y4 = heightMap->data[rightx][rightz];
//
//	float yLeftx = zRatio * y2 + (1.0f - zRatio) * y1;
//	float yRightx = zRatio * y4 + (1.0f - zRatio) * y3;
//
//	float y = yRightx * xRatio + (1.0f - xRatio) * yLeftx;
//
//	//y = 5.0f; //////////////////////////////////////////////////////// FORCE Y to dry adiabat
//
//	particlePositions.push_back(glm::vec3(randx, y, randz));
//
//
//	mapFromSimulationBox(y);
//
//	Particle p;
//	p.position = glm::vec3(randx, y, randz);
//	p.velocity = glm::vec3(0.0f);
//
//
//	if (profileMap && profileMap->height >= heightMap->height && profileMap->width >= heightMap->width) {
//
//		glm::vec2 p1 = profileMap->data[leftx][leftz];
//		glm::vec2 p2 = profileMap->data[leftx][rightz];
//		glm::vec2 p3 = profileMap->data[rightx][leftz];
//		glm::vec2 p4 = profileMap->data[rightx][rightz];
//
//		glm::vec2 pi1 = zRatio * p2 + (1.0f - zRatio) * p1;
//		glm::vec2 pi2 = zRatio * p4 + (1.0f - zRatio) * p3;
//
//		glm::vec2 pif = xRatio * pi2 + (1.0f - xRatio) * pi1;
//		glm::ivec2 pii = (glm::ivec2)pif;
//
//		if (pii.y != pii.x) {
//			p.profileIndex = (rand() % (pii.y - pii.x) + pii.x) % (stlpDiagram->numProfiles - 1);
//		} else {
//			p.profileIndex = pii.x % (stlpDiagram->numProfiles - 1);
//		}
//
//	} else {
//		p.profileIndex = rand() % (stlpDiagram->numProfiles - 1);
//	}
//
//
//	p.updatePressureVal();
//
//	particles.push_back(p);
//	numParticles++;
//
//}
