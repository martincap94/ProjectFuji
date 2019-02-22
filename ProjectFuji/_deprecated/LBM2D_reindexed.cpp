#include "LBM2D_reindexed.h"

#include <vector>
#include <iostream>

#include <glm/gtx/string_cast.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


//__global__ void collisionStepKernel(Node *backLattice) {
	//float weightMiddle = 4.0f / 9.0f;
	//float weightAxis = 1.0f / 9.0f;
	//float weightDiagonal = 1.0f / 36.0f;

	//float macroDensity = calculateMacroscopicDensity(x, y);

	//glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);


	//velocities[x][y] = glm::vec2(macroVelocity.x, macroVelocity.y);


	//// let's find the equilibrium
	//float leftTermMiddle = weightMiddle * macroDensity;
	//float leftTermAxis = weightAxis * macroDensity;
	//float leftTermDiagonal = weightDiagonal * macroDensity;

	//// optimize these operations later

	//float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	//float thirdTerm = 1.5f * macroVelocityDot;

	//float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	//// this can all be rewritten into arrays + for cycles!
	//float dotProd = glm::dot(vRight, macroVelocity);
	//float firstTerm = 3.0f * dotProd;
	//float secondTerm = 4.5f * dotProd * dotProd;
	//float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	//dotProd = glm::dot(vTop, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	//dotProd = glm::dot(vLeft, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	//dotProd = glm::dot(vBottom, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	//dotProd = glm::dot(vTopRight, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	//dotProd = glm::dot(vTopLeft, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	//dotProd = glm::dot(vBottomLeft, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	//dotProd = glm::dot(vBottomRight, macroVelocity);
	//firstTerm = 3.0f * dotProd;
	//secondTerm = 4.5f * dotProd * dotProd;
	//float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);

	//backLattice[x][y].adj[DIR_MIDDLE] -= ITAU * (backLattice[x][y].adj[DIR_MIDDLE] - middleEq);
	//backLattice[x][y].adj[DIR_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_RIGHT] - rightEq);
	//backLattice[x][y].adj[DIR_TOP] -= ITAU * (backLattice[x][y].adj[DIR_TOP] - topEq);
	//backLattice[x][y].adj[DIR_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_LEFT] - leftEq);
	//backLattice[x][y].adj[DIR_TOP_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_TOP_RIGHT] - topRightEq);
	//backLattice[x][y].adj[DIR_TOP_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_TOP_LEFT] - topLeftEq);
	//backLattice[x][y].adj[DIR_BOTTOM_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
	//backLattice[x][y].adj[DIR_BOTTOM_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


	//for (int i = 0; i < 9; i++) {
	//	if (backLattice[x][y].adj[i] < 0.0f) {
	//		backLattice[x][y].adj[i] = 0.0f;
	//	} else if (backLattice[x][y].adj[i] > 1.0f) {
	//		backLattice[x][y].adj[i] = 1.0f;
	//	}
	//}

//}


void LBM2D_reindexed::collisionStepCUDA() {

	cudaMemcpy2D(d_backLattice, backLatticePitch, backLattice, GRID_WIDTH, GRID_WIDTH, GRID_HEIGHT, cudaMemcpyHostToDevice);



}

LBM2D_reindexed::LBM2D_reindexed() {
}

LBM2D_reindexed::LBM2D_reindexed(ParticleSystem *particleSystem) : particleSystem(particleSystem) {

	particleVertices = particleSystem->particleVertices;
	frontLattice = new Node*[GRID_WIDTH]();
	backLattice = new Node*[GRID_WIDTH]();
	velocities = new glm::vec2*[GRID_WIDTH]();

	//cudaMallocPitch((void**)&d_frontLattice, &frontLatticePitch, GRID_WIDTH, GRID_HEIGHT);



	for (int x = 0; x < GRID_WIDTH; x++) {
		frontLattice[x] = new Node[GRID_HEIGHT]();
		backLattice[x] = new Node[GRID_HEIGHT]();
		velocities[x] = new glm::vec2[GRID_HEIGHT]();
	}
	initTestCollider();

	initBuffers();
	initLattice();

}


LBM2D_reindexed::~LBM2D_reindexed() {
	for (int x = 0; x < GRID_WIDTH; x++) {
		delete[] frontLattice[x];
		delete[] backLattice[x];
		delete[] velocities[x];
	}
	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;

	delete tCol;

	cudaFree(d_frontLattice);
}

void LBM2D_reindexed::draw(ShaderProgram &shader) {
	glPointSize(1.0f);
	shader.setVec3("color", glm::vec3(0.4f, 0.4f, 0.1f));
	glUseProgram(shader.id);

	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, GRID_WIDTH * GRID_HEIGHT);


	//cout << "Velocity arrows size = " << velocityArrows.size() << endl;

#ifdef DRAW_VELOCITY_ARROWS
	shader.setVec3("color", glm::vec3(0.2f, 0.3f, 1.0f));
	glBindVertexArray(velocityVAO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * velocityArrows.size(), &velocityArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, velocityArrows.size());
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	shader.setVec3("color", glm::vec3(0.8f, 1.0f, 0.6f));

	glBindVertexArray(particleArrowsVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particleArrows.size(), &particleArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, particleArrows.size());
#endif

	// Draw test collider
	tCol->draw(shader);





}

void LBM2D_reindexed::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();



	//collisionStepCUDA();
	collisionStep();
	moveParticles();

	swapLattices();


}

void LBM2D_reindexed::clearBackLattice() {
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int i = 0; i < 9; i++) {
				backLattice[x][y].adj[i] = 0.0f;
			}
		}
	}
	velocityArrows.clear();
	particleArrows.clear();
}

void LBM2D_reindexed::streamingStep() {

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			backLattice[x][y].adj[DIR_MIDDLE] += frontLattice[x][y].adj[DIR_MIDDLE];

			int right;
			int left;
			int top;
			int bottom;

			right = x + 1;
			left = x - 1;
			top = y + 1;
			bottom = y - 1;
			if (right > GRID_WIDTH - 1) {
				right = GRID_WIDTH - 1;
			}
			if (left < 0) {
				left = 0;
			}
			if (top > GRID_HEIGHT - 1) {
				top = GRID_HEIGHT - 1;
			}
			if (bottom < 0) {
				bottom = 0;
			}


			backLattice[x][y].adj[DIR_RIGHT] += frontLattice[left][y].adj[DIR_RIGHT];
			backLattice[x][y].adj[DIR_TOP] += frontLattice[x][bottom].adj[DIR_TOP];
			backLattice[x][y].adj[DIR_LEFT] += frontLattice[right][y].adj[DIR_LEFT];
			backLattice[x][y].adj[DIR_BOTTOM] += frontLattice[x][top].adj[DIR_BOTTOM];
			backLattice[x][y].adj[DIR_TOP_RIGHT] += frontLattice[left][bottom].adj[DIR_TOP_RIGHT];
			backLattice[x][y].adj[DIR_TOP_LEFT] += frontLattice[right][bottom].adj[DIR_TOP_LEFT];
			backLattice[x][y].adj[DIR_BOTTOM_LEFT] += frontLattice[right][top].adj[DIR_BOTTOM_LEFT];
			backLattice[x][y].adj[DIR_BOTTOM_RIGHT] += frontLattice[left][top].adj[DIR_BOTTOM_RIGHT];

			for (int i = 0; i < 9; i++) {
				if (backLattice[x][y].adj[i] < 0.0f) {
					backLattice[x][y].adj[i] = 0.0f;
				} else if (backLattice[x][y].adj[i] > 1.0f) {
					backLattice[x][y].adj[i] = 1.0f;
				}
			}

		}
	}

}

void LBM2D_reindexed::collisionStep() {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			/*if (x == 0 || x == GRID_WIDTH - 1) {
				continue;
			}
			if (y == 0 || y == GRID_HEIGHT - 1) {
				continue;
			}*/


			float macroDensity = calculateMacroscopicDensity(x, y);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);


			velocities[x][y] = glm::vec2(macroVelocity.x, macroVelocity.y);


			velocityArrows.push_back(glm::vec3(x, y, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[x][y] * 5.0f, -1.0f) + glm::vec3(x, y, 0.0f));



			// let's find the equilibrium
			float leftTermMiddle = weightMiddle * macroDensity;
			float leftTermAxis = weightAxis * macroDensity;
			float leftTermDiagonal = weightDiagonal * macroDensity;

			// optimize these operations later

			float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
			float thirdTerm = 1.5f * macroVelocityDot;

			float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

			// this can all be rewritten into arrays + for cycles!
			float dotProd = glm::dot(vRight, macroVelocity);
			float firstTerm = 3.0f * dotProd;
			float secondTerm = 4.5f * dotProd * dotProd;
			float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

			dotProd = glm::dot(vTop, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

			dotProd = glm::dot(vLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottom, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vTopRight, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vTopLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottomLeft, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


			dotProd = glm::dot(vBottomRight, macroVelocity);
			firstTerm = 3.0f * dotProd;
			secondTerm = 4.5f * dotProd * dotProd;
			float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);

			backLattice[x][y].adj[DIR_MIDDLE] -= ITAU * (backLattice[x][y].adj[DIR_MIDDLE] - middleEq);
			backLattice[x][y].adj[DIR_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_RIGHT] - rightEq);
			backLattice[x][y].adj[DIR_TOP] -= ITAU * (backLattice[x][y].adj[DIR_TOP] - topEq);
			backLattice[x][y].adj[DIR_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_LEFT] - leftEq);
			backLattice[x][y].adj[DIR_TOP_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[x][y].adj[DIR_TOP_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[x][y].adj[DIR_BOTTOM_LEFT] -= ITAU * (backLattice[x][y].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[x][y].adj[DIR_BOTTOM_RIGHT] -= ITAU * (backLattice[x][y].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[x][y].adj[i] < 0.0f) {
					backLattice[x][y].adj[i] = 0.0f;
				} else if (backLattice[x][y].adj[i] > 1.0f) {
					backLattice[x][y].adj[i] = 1.0f;
				}
			}

		}
	}


}


void LBM2D_reindexed::moveParticles() {


	glm::vec2 adjVelocities[4];
	for (int i = 0; i < particleSystem->numParticles; i++) {
		float x = particleVertices[i].x;
		float y = particleVertices[i].y;


		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[leftX][topY];
		adjVelocities[1] = velocities[rightX][topY];
		adjVelocities[2] = velocities[leftX][bottomY];
		adjVelocities[3] = velocities[rightX][bottomY];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);

#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		particleArrows.push_back(particleVertices[i]);
#endif
		particleVertices[i] += glm::vec3(finalVelocity, 0.0f);
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		glm::vec3 tmp = particleVertices[i] + 10.0f * glm::vec3(finalVelocity, 0.0f);
		particleArrows.push_back(tmp);
#endif


		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= GRID_HEIGHT - 1) {
			particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
			if (respawnIndex >= GRID_HEIGHT - 1) {
				respawnIndex = 0;
			}
		}

	}


}

void LBM2D_reindexed::updateInlets() {


	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = glm::vec3(1.0f, 0.0f, 0.0f);

	// let's find the equilibrium
	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermDiagonal = weightDiagonal * macroDensity;

	// optimize these operations later

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	// this can all be rewritten into arrays + for cycles!
	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	for (int y = 0; y < GRID_HEIGHT; y++) {
		backLattice[0][y].adj[DIR_MIDDLE] = middleEq;
		backLattice[0][y].adj[DIR_RIGHT] = rightEq;
		backLattice[0][y].adj[DIR_TOP] = topEq;
		backLattice[0][y].adj[DIR_LEFT] = leftEq;
		backLattice[0][y].adj[DIR_TOP_RIGHT] = topRightEq;
		backLattice[0][y].adj[DIR_TOP_LEFT] = topLeftEq;
		backLattice[0][y].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		backLattice[0][y].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (backLattice[0][y].adj[i] < 0.0f) {
				backLattice[0][y].adj[i] = 0.0f;
			} else if (backLattice[0][y].adj[i] > 1.0f) {
				backLattice[0][y].adj[i] = 1.0f;
			}
		}
		velocities[0][y] = macroVelocity;
	}




}

void LBM2D_reindexed::updateColliders() {

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			if (/*testCollider[row][col] ||*/ y == 0 || y == GRID_HEIGHT - 1 || tCol->area[x + GRID_WIDTH * y]) {

				float right = backLattice[x][y].adj[DIR_RIGHT];
				float top = backLattice[x][y].adj[DIR_TOP];
				float left = backLattice[x][y].adj[DIR_LEFT];
				float bottom = backLattice[x][y].adj[DIR_BOTTOM];
				float topRight = backLattice[x][y].adj[DIR_TOP_RIGHT];
				float topLeft = backLattice[x][y].adj[DIR_TOP_LEFT];
				float bottomLeft = backLattice[x][y].adj[DIR_BOTTOM_LEFT];
				float bottomRight = backLattice[x][y].adj[DIR_BOTTOM_RIGHT];
				backLattice[x][y].adj[DIR_RIGHT] = left;
				backLattice[x][y].adj[DIR_TOP] = bottom;
				backLattice[x][y].adj[DIR_LEFT] = right;
				backLattice[x][y].adj[DIR_BOTTOM] = top;
				backLattice[x][y].adj[DIR_TOP_RIGHT] = bottomLeft;
				backLattice[x][y].adj[DIR_TOP_LEFT] = bottomRight;
				backLattice[x][y].adj[DIR_BOTTOM_LEFT] = topRight;
				backLattice[x][y].adj[DIR_BOTTOM_RIGHT] = topLeft;


				float macroDensity = calculateMacroscopicDensity(x, y);
				glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);
				velocities[x][y] = macroVelocity;

			}


		}
	}

}


void LBM2D_reindexed::initBuffers() {


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	vector<glm::vec3> bData;
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			bData.push_back(glm::vec3(x, y, 0.0f));
		}
	}

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


	// Velocity arrows
	glGenVertexArrays(1, &velocityVAO);
	glBindVertexArray(velocityVAO);
	glGenBuffers(1, &velocityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);




	// Particle arrows
	glGenVertexArrays(1, &particleArrowsVAO);
	glBindVertexArray(particleArrowsVAO);
	glGenBuffers(1, &particleArrowsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);


	glBindVertexArray(0);


}

void LBM2D_reindexed::initLattice() {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			frontLattice[x][y].adj[DIR_MIDDLE] = weightMiddle;
			for (int dir = 1; dir <= 4; dir++) {
				frontLattice[x][y].adj[dir] = weightAxis;
			}
			for (int dir = 5; dir <= 8; dir++) {
				frontLattice[x][y].adj[dir] = weightDiagonal;
			}
		}
	}


}

void LBM2D_reindexed::initTestCollider() {
	tCol = new LatticeCollider(COLLIDER_FILENAME);

}

void LBM2D_reindexed::swapLattices() {
	Node **tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;
}

float LBM2D_reindexed::calculateMacroscopicDensity(int x, int y) {

	float macroDensity = 0.0f;
	for (int i = 0; i < 9; i++) {
		macroDensity += backLattice[x][y].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM2D_reindexed::calculateMacroscopicVelocity(int x, int y, float macroDensity) {
	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	macroVelocity += vRight * backLattice[x][y].adj[DIR_RIGHT];
	macroVelocity += vTop * backLattice[x][y].adj[DIR_TOP];
	macroVelocity += vLeft * backLattice[x][y].adj[DIR_LEFT];
	macroVelocity += vBottom * backLattice[x][y].adj[DIR_BOTTOM];
	macroVelocity += vTopRight * backLattice[x][y].adj[DIR_TOP_RIGHT];
	macroVelocity += vTopLeft * backLattice[x][y].adj[DIR_TOP_LEFT];
	macroVelocity += vBottomLeft * backLattice[x][y].adj[DIR_BOTTOM_LEFT];
	macroVelocity += vBottomRight * backLattice[x][y].adj[DIR_BOTTOM_RIGHT];
	macroVelocity /= macroDensity;


	return macroVelocity;
}
