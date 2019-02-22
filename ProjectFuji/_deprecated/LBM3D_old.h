#pragma once

#include "Config.h"
#include "ParticleSystem.h"
#include "DataStructures.h"

#include <vector>

class LBM3D {

	/*struct Node3D {
		float adj[19];
	};*/

	// 3rd ordering
	/*enum EDirection3D {
		DIR_MIDDLE = 0,
		DIR_RIGHT_FACE,
		DIR_LEFT_FACE,
		DIR_BACK_FACE,
		DIR_FRONT_FACE,
		DIR_TOP_FACE,
		DIR_BOTTOM_FACE,
		DIR_BACK_RIGHT_EDGE,
		DIR_BACK_LEFT_EDGE,
		DIR_FRONT_RIGHT_EDGE,
		DIR_FRONT_LEFT_EDGE,
		DIR_TOP_BACK_EDGE,
		DIR_TOP_FRONT_EDGE,
		DIR_BOTTOM_BACK_EDGE,
		DIR_BOTTOM_FRONT_EDGE,
		DIR_TOP_RIGHT_EDGE,
		DIR_TOP_LEFT_EDGE,
		DIR_BOTTOM_RIGHT_EDGE,
		DIR_BOTTOM_LEFT_EDGE
	};*/

	const glm::vec3 vMiddle = glm::vec3(0.0f, 0.0f, 0.0f);
	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBack = glm::vec3(0.0f, 0.0f, -1.0f);
	const glm::vec3 vFront = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vBackRight = glm::vec3(1.0f, 0.0f, -1.0f);
	const glm::vec3 vBackLeft = glm::vec3(-1.0f, 0.0f, -1.0f);
	const glm::vec3 vFrontRight = glm::vec3(1.0f, 0.0f, 1.0f);
	const glm::vec3 vFrontLeft = glm::vec3(-1.0f, 0.0f, 1.0f);
	const glm::vec3 vTopBack = glm::vec3(0.0f, 1.0f, -1.0f);
	const glm::vec3 vTopFront = glm::vec3(0.0f, 1.0f, 1.0f);
	const glm::vec3 vBottomBack = glm::vec3(0.0f, -1.0f, -1.0f);
	const glm::vec3 vBottomFront = glm::vec3(0.0f, -1.0f, 1.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);



public:


	Node3D ***frontLattice;
	Node3D ***backLattice;

	ParticleSystem *particleSystem;
	glm::vec3 *particleVertices;

	glm::vec3 ***velocities;

	bool ***testCollider;
	vector<glm::vec3> colliderVertices;


	LBM3D();
	LBM3D(ParticleSystem *particleSystem);
	~LBM3D();

	void draw(ShaderProgram &shader);

	void doStep();
	void clearBackLattice();
	void streamingStep();
	void collisionStep();
	void moveParticles();
	void updateInlets();
	void updateColliders();

private:

	int respawnY = 0;
	int respawnZ = 0;

	GLuint colliderVAO;
	GLuint colliderVBO;

	GLuint velocityVBO;
	GLuint velocityVAO;

	GLuint particleArrowsVAO;
	GLuint particleArrowsVBO;

	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	void initBuffers();
	void initLattice();
	void initColliders();

	void swapLattices();

	float calculateMacroscopicDensity(int x, int y, int z);
	glm::vec3 calculateMacroscopicVelocity(int x, int y, int z, float macroDensity);


};

