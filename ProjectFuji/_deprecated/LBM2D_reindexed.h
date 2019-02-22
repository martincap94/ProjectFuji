#pragma once

#include "Config.h"

#include "ShaderProgram.h"
#include "ParticleSystem.h"
#include "LatticeCollider.h"

#include <vector>

#include "LBM.h"

class LBM2D_reindexed : public LBM {
	
	/*struct Node {
		float adj[9];
	};*/

	struct Node {
		float adj[9];
	};


	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

	enum EDirection {
		DIR_MIDDLE = 0,
		DIR_RIGHT,
		DIR_TOP,
		DIR_LEFT,
		DIR_BOTTOM,
		DIR_TOP_RIGHT,
		DIR_TOP_LEFT,
		DIR_BOTTOM_LEFT,
		DIR_BOTTOM_RIGHT
	};


public:

	Node **frontLattice;
	Node **backLattice;

	size_t frontLatticePitch;
	Node *d_frontLattice;
	size_t backLatticePitch;
	Node *d_backLattice;

	ParticleSystem *particleSystem;
	glm::vec3 *particleVertices;

	//bool **testCollider;
	//glm::vec3 testColliderExtent[2];

	LatticeCollider *tCol;


	//GLuint tcVAO;
	//GLuint tcVBO;

	glm::vec2 **velocities;
	vector<glm::vec3> velocityArrows;
	vector<glm::vec3> particleArrows;

	LBM2D_reindexed();
	LBM2D_reindexed(ParticleSystem *particleSystem);
	~LBM2D_reindexed();

	void draw(ShaderProgram &shader);

	void doStep();
	void clearBackLattice();
	void streamingStep();
	void collisionStep();
	void collisionStepCUDA();

	void moveParticles();
	void updateInlets();
	void updateColliders();

private:

	GLuint vbo;
	GLuint vao;

	GLuint velocityVBO;
	GLuint velocityVAO;

	GLuint particleArrowsVAO;
	GLuint particleArrowsVBO;

	int respawnIndex = 0;

	void initBuffers();

	void initLattice();
	void initTestCollider();


	void swapLattices();
	float calculateMacroscopicDensity(int x, int y);
	glm::vec3 calculateMacroscopicVelocity(int x, int y, float macroDensity);



};

