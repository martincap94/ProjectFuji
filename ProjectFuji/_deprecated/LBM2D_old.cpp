#include "LBM2D.h"

#include <vector>
#include <iostream>

#include <glm/gtx/string_cast.hpp>

LBM2D::LBM2D() {
}

LBM2D::LBM2D(int width, int height, ParticleSystem *particleSystem) : width(width), height(height), particleSystem(particleSystem) {

	particleVertices = particleSystem->particleVertices;
	frontLattice = new Node*[width];
	backLattice = new Node*[width];
	velocities = new glm::vec2*[width]();

	for (int row = 0; row < height; row++) {
		frontLattice[row] = new Node[height];
		backLattice[row] = new Node[height];
		velocities[row] = new glm::vec2[height]();
	}
	initTestCollider();

	initBuffers();
	initLattice();

}


LBM2D::~LBM2D() {
	for (int row = 0; row < height; row++) {
		delete[] frontLattice[row];
		delete[] backLattice[row];
		delete[] velocities[row];
	}
	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;

	delete tCol;
}

void LBM2D::draw(ShaderProgram &shader) {
	glPointSize(1.0f);
	shader.setVec3("color", glm::vec3(0.4f, 0.4f, 0.1f));
	glUseProgram(shader.id);

	glBindVertexArray(vao);
	glDrawArrays(GL_POINTS, 0, width * height);


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

void LBM2D::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();
	collisionStep();
	moveParticles();

	swapLattices();


}

void LBM2D::clearBackLattice() {
	for (int row = 0; row < GRID_WIDTH; row++) {
		for (int col = 0; col < GRID_HEIGHT; col++) {
			for (int i = 0; i < 9; i++) {
				backLattice[row][col].adj[i] = 0.0f;
			}
		}
	}
	velocityArrows.clear();
	particleArrows.clear();
}

void LBM2D::streamingStep() {

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {

			backLattice[row][col].adj[DIR_MIDDLE] += frontLattice[row][col].adj[DIR_MIDDLE];

			int right, top, left, bottom;
			right = col + 1;
			top = row + 1;
			left = col - 1;
			bottom = row - 1;
			if (bottom < 0) {
				bottom = 0;
			}
			if (top > GRID_HEIGHT - 1) {
				top = GRID_HEIGHT - 1;
			}
			if (left < 0) {
				left = 0;
			}
			if (right > GRID_WIDTH - 1) {
				right = GRID_WIDTH - 1;
			}


			backLattice[row][col].adj[DIR_RIGHT] += frontLattice[row][left].adj[DIR_RIGHT];
			backLattice[row][col].adj[DIR_TOP] += frontLattice[bottom][col].adj[DIR_TOP];
			backLattice[row][col].adj[DIR_LEFT] += frontLattice[row][right].adj[DIR_LEFT];
			backLattice[row][col].adj[DIR_BOTTOM] += frontLattice[top][col].adj[DIR_BOTTOM];
			backLattice[row][col].adj[DIR_TOP_RIGHT] += frontLattice[bottom][left].adj[DIR_TOP_RIGHT];
			backLattice[row][col].adj[DIR_TOP_LEFT] += frontLattice[bottom][right].adj[DIR_TOP_LEFT];
			backLattice[row][col].adj[DIR_BOTTOM_LEFT] += frontLattice[top][right].adj[DIR_BOTTOM_LEFT];
			backLattice[row][col].adj[DIR_BOTTOM_RIGHT] += frontLattice[top][left].adj[DIR_BOTTOM_RIGHT];

		}
	}

}

void LBM2D::collisionStep() {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int row = 0; row < GRID_HEIGHT; row++) {
		for (int col = 0; col < GRID_WIDTH; col++) {

			/*if (row == 0 || row == GRID_HEIGHT - 1) {
				continue;
			}*/
			/*if (col == 0 || col == GRID_WIDTH - 1) {
				continue;
			}*/

			

			float macroDensity = calculateMacroscopicDensity(nullptr, row, col);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(row, col, macroDensity);


			velocities[row][col] = glm::vec2(macroVelocity.x, macroVelocity.y);


			velocityArrows.push_back(glm::vec3(col, row, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[row][col] * 5.0f, -1.0f) + glm::vec3(col, row, 0.0f));



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

			backLattice[row][col].adj[DIR_MIDDLE] -= ITAU * (backLattice[row][col].adj[DIR_MIDDLE] - middleEq);
			backLattice[row][col].adj[DIR_RIGHT] -= ITAU * (backLattice[row][col].adj[DIR_RIGHT] - rightEq);
			backLattice[row][col].adj[DIR_TOP] -= ITAU * (backLattice[row][col].adj[DIR_TOP] - topEq);
			backLattice[row][col].adj[DIR_LEFT] -= ITAU * (backLattice[row][col].adj[DIR_LEFT] - leftEq);
			backLattice[row][col].adj[DIR_TOP_RIGHT] -= ITAU * (backLattice[row][col].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[row][col].adj[DIR_TOP_LEFT] -= ITAU * (backLattice[row][col].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[row][col].adj[DIR_BOTTOM_LEFT] -= ITAU * (backLattice[row][col].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[row][col].adj[DIR_BOTTOM_RIGHT] -= ITAU * (backLattice[row][col].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[row][col].adj[i] < 0.0f) {
					backLattice[row][col].adj[i] = 0.0f;
				} else if (backLattice[row][col].adj[i] > 1.0f) {
					backLattice[row][col].adj[i] = 1.0f;
				}
			}

		}
	}


}

void LBM2D::moveParticles() {


	glm::vec2 adjVelocities[4];
	for (int i = 0; i < particleSystem->numParticles; i++) {
		float x = particleVertices[i].x;
		float y = particleVertices[i].y;

		int leftCol = (int)x;
		int rightCol = leftCol + 1;
		int bottomRow = (int)y;
		int topRow = bottomRow + 1;

		float horizontalRatio = x - floor(x);
		float verticalRatio = y - floor(y);

		adjVelocities[0] = velocities[topRow][rightCol];
		adjVelocities[1] = velocities[topRow][leftCol];
		adjVelocities[2] = velocities[bottomRow][leftCol];
		adjVelocities[3] = velocities[bottomRow][rightCol];

		glm::vec2 topVelocity = adjVelocities[0] * (1 - horizontalRatio) + adjVelocities[1] * horizontalRatio;
		glm::vec2 bottomVelocity = adjVelocities[2] * (1 - horizontalRatio) + adjVelocities[3] * horizontalRatio;

		glm::vec2 finalVelocity = (1 - verticalRatio) * bottomVelocity + verticalRatio * topVelocity;

		particleArrows.push_back(particleVertices[i]);

		particleVertices[i] += glm::vec3(finalVelocity, 0.0f);

		glm::vec3 tmp = particleVertices[i] + 10.0f * glm::vec3(finalVelocity, 0.0f);

		particleArrows.push_back(tmp);


		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= GRID_HEIGHT - 1) {
			//cout << "Problem??? :)" << endl;
			//particleVertices[i] = glm::vec3(rand() % (GRID_WIDTH - 1), rand() % (GRID_HEIGHT - 1), 0.0f);
			particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
			if (respawnIndex >= GRID_HEIGHT - 1) {
				respawnIndex = 0;
			}
		}

	}


}

void LBM2D::updateInlets() {


	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = glm::vec3(0.4f, 0.0f, 0.0f);

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


	for (int row = 0; row < GRID_HEIGHT; row++) {
		backLattice[row][0].adj[DIR_MIDDLE] = middleEq;
		backLattice[row][0].adj[DIR_RIGHT] = rightEq;
		backLattice[row][0].adj[DIR_TOP] = topEq;
		backLattice[row][0].adj[DIR_LEFT] = leftEq;
		backLattice[row][0].adj[DIR_TOP_RIGHT] = topRightEq;
		backLattice[row][0].adj[DIR_TOP_LEFT] = topLeftEq;
		backLattice[row][0].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		backLattice[row][0].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (backLattice[row][0].adj[i] < 0.0f) {
				backLattice[row][0].adj[i] = 0.0f;
			} else if (backLattice[row][0].adj[i] > 1.0f) {
				backLattice[row][0].adj[i] = 1.0f;
			}
		}
		velocities[row][0] = macroVelocity;
	}


	

}

void LBM2D::updateColliders() {

	for (int row = 0; row < GRID_HEIGHT; row++) {
		for (int col = 0; col < GRID_WIDTH; col++) {

			if (/*testCollider[row][col] ||*/ row == 0 || row == GRID_HEIGHT - 1 || tCol->area[col + row * GRID_WIDTH]) {

				float right = backLattice[row][col].adj[DIR_RIGHT];
				float top = backLattice[row][col].adj[DIR_TOP];
				float left = backLattice[row][col].adj[DIR_LEFT];
				float bottom = backLattice[row][col].adj[DIR_BOTTOM];
				float topRight = backLattice[row][col].adj[DIR_TOP_RIGHT];
				float topLeft = backLattice[row][col].adj[DIR_TOP_LEFT];
				float bottomLeft = backLattice[row][col].adj[DIR_BOTTOM_LEFT];
				float bottomRight = backLattice[row][col].adj[DIR_BOTTOM_RIGHT];
				backLattice[row][col].adj[DIR_RIGHT] = left;
				backLattice[row][col].adj[DIR_TOP] = bottom;
				backLattice[row][col].adj[DIR_LEFT] = right;
				backLattice[row][col].adj[DIR_BOTTOM] = top;
				backLattice[row][col].adj[DIR_TOP_RIGHT] = bottomLeft;
				backLattice[row][col].adj[DIR_TOP_LEFT] = bottomRight;
				backLattice[row][col].adj[DIR_BOTTOM_LEFT] = topRight;
				backLattice[row][col].adj[DIR_BOTTOM_RIGHT] = topLeft;


				float macroDensity = calculateMacroscopicDensity(nullptr, row, col);
				glm::vec3 macroVelocity = calculateMacroscopicVelocity(row, col, macroDensity);
				velocities[row][col] = macroVelocity;

			}


		}
	}

}


void LBM2D::initBuffers() {


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	vector<glm::vec3> bData;
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			bData.push_back(glm::vec3(col, row, 0.0f));
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

void LBM2D::initLattice() {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int row = 0; row < GRID_HEIGHT; row++) {
		for (int col = 0; col < GRID_WIDTH; col++) {
			frontLattice[row][col].adj[DIR_MIDDLE] = weightMiddle;
			for (int dir = 1; dir <= 4; dir++) {
				frontLattice[row][col].adj[dir] = weightAxis;
			}
			for (int dir = 5; dir <= 8; dir++) {
				frontLattice[row][col].adj[dir] = weightDiagonal;
			}
		}
	}


}

void LBM2D::initTestCollider() {
	//testColliderExtent[0] = glm::vec3(60, 25, 0);
	//testColliderExtent[1] = glm::vec3(61, 60, 0);
	///*for (int row = testColliderExtent[0].y; row <= testColliderExtent[1].y; row++) {
	//	for (int col = testColliderExtent[0].x; col <= testColliderExtent[1].x; col++) {
	//		testCollider[row][col] = true;
	//	}
	//}*/
	//for (int row = testColliderExtent[0].y; row <= testColliderExtent[1].y; row++) {
	//	testCollider[row][(int)testColliderExtent[0].x] = true;
	//	testCollider[row][(int)testColliderExtent[1].x] = true;
	//}
	//for (int col = testColliderExtent[0].x; col <= testColliderExtent[1].x; col++) {
	//	testCollider[(int)testColliderExtent[0].y][col] = true;
	//	testCollider[(int)testColliderExtent[1].y][col] = true;
	//}

	tCol = new LatticeCollider(COLLIDER_FILENAME);

}

void LBM2D::swapLattices() {
	Node **tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;
}

float LBM2D::calculateMacroscopicDensity(Node **lattice, int row, int col) {

	float macroDensity = 0.0f;
	for (int i = 0; i < 9; i++) {
		macroDensity += backLattice[row][col].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM2D::calculateMacroscopicVelocity(int row, int col, float macroDensity) {
	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	macroVelocity += vRight * backLattice[row][col].adj[DIR_RIGHT];
	macroVelocity += vTop * backLattice[row][col].adj[DIR_TOP];
	macroVelocity += vLeft * backLattice[row][col].adj[DIR_LEFT];
	macroVelocity += vBottom * backLattice[row][col].adj[DIR_BOTTOM];
	macroVelocity += vTopRight * backLattice[row][col].adj[DIR_TOP_RIGHT];
	macroVelocity += vTopLeft * backLattice[row][col].adj[DIR_TOP_LEFT];
	macroVelocity += vBottomLeft * backLattice[row][col].adj[DIR_BOTTOM_LEFT];
	macroVelocity += vBottomRight * backLattice[row][col].adj[DIR_BOTTOM_RIGHT];
	macroVelocity /= macroDensity;


	return macroVelocity;
}
