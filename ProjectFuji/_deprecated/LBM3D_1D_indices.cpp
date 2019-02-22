#include "LBM3D_1D_indices.h"

#define GRID_SIZE GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH

LBM3D_1D_indices::LBM3D_1D_indices() {
}

LBM3D_1D_indices::LBM3D_1D_indices(Particles * particleSystem) : particleSystem(particleSystem) {

	particleVertices = particleSystem->particleVertices;

	frontLattice = new Node[GRID_SIZE]();
	backLattice = new Node[GRID_SIZE]();
	velocities = new glm::vec3**[GRID_WIDTH]();
	sphereCollider = new bool**[GRID_WIDTH]();

	for (int i = 0; i < GRID_WIDTH; i++) {
		velocities[i] = new glm::vec3*[GRID_HEIGHT]();
		sphereCollider[i] = new bool*[GRID_HEIGHT]();

		for (int j = 0; j < GRID_HEIGHT; j++) {
			velocities[i][j] = new glm::vec3[GRID_DEPTH]();
			sphereCollider[i][j] = new bool[GRID_DEPTH]();
		}
	}


	initColliders();

	initBuffers();
	initLattice();

}


LBM3D_1D_indices::~LBM3D_1D_indices() {
	for (int i = 0; i < GRID_WIDTH; i++) {
		for (int j = 0; j < GRID_HEIGHT; j++) {
			delete[] velocities[i][j];
			delete[] sphereCollider[i][j];

		}
		delete[] velocities[i];
		delete[] sphereCollider[i];

	}
	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;
	delete[] sphereCollider;



}

void LBM3D_1D_indices::draw(ShaderProgram & shader) {

	glUseProgram(shader.id);
	glBindVertexArray(colliderVAO);

	glPointSize(8.0f);
	shader.setVec3("color", glm::vec3(1.0f, 1.0f, 0.4f));

	glDrawArrays(GL_POINTS, 0, colliderVertices.size());


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



}

void LBM3D_1D_indices::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();
	collisionStep();
	moveParticles();

	swapLattices();
}

void LBM3D_1D_indices::clearBackLattice() {
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int z = 0; z < GRID_DEPTH; z++) {
				int idx = getIdx(x, y, z);
				for (int i = 0; i < 19; i++) {
					backLattice[idx].adj[i] = 0.0f;
				}
			}
		}
	}
#ifdef DRAW_VELOCITY_ARROWS
	velocityArrows.clear();
#endif
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	particleArrows.clear();
#endif

}

void LBM3D_1D_indices::streamingStep() {


	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int z = 0; z < GRID_DEPTH; z++) {
				int idx = getIdx(x, y, z);
				backLattice[idx].adj[DIR_MIDDLE] += frontLattice[idx].adj[DIR_MIDDLE];

				int right;
				int left;
				int top;
				int bottom;
				int front;
				int back;

				right = x + 1;
				left = x - 1;
				top = y + 1;
				bottom = y - 1;
				front = z + 1;
				back = z - 1;
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
				if (front > GRID_DEPTH - 1) {
					front = GRID_DEPTH - 1;
				}
				if (back < 0) {
					back = 0;
				}

				backLattice[idx].adj[DIR_LEFT_FACE] += frontLattice[getIdx(right, y, z)].adj[DIR_LEFT_FACE];
				backLattice[idx].adj[DIR_FRONT_FACE] += frontLattice[getIdx(x, y, back)].adj[DIR_FRONT_FACE];
				backLattice[idx].adj[DIR_BOTTOM_FACE] += frontLattice[getIdx(x, top, z)].adj[DIR_BOTTOM_FACE];
				backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] += frontLattice[getIdx(right, y, back)].adj[DIR_FRONT_LEFT_EDGE];
				backLattice[idx].adj[DIR_BACK_LEFT_EDGE] += frontLattice[getIdx(right, y, front)].adj[DIR_BACK_LEFT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] += frontLattice[getIdx(right, top, z)].adj[DIR_BOTTOM_LEFT_EDGE];
				backLattice[idx].adj[DIR_TOP_LEFT_EDGE] += frontLattice[getIdx(right, bottom, z)].adj[DIR_TOP_LEFT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] += frontLattice[getIdx(x, top, back)].adj[DIR_BOTTOM_FRONT_EDGE];
				backLattice[idx].adj[DIR_TOP_FRONT_EDGE] += frontLattice[getIdx(x, bottom, back)].adj[DIR_TOP_FRONT_EDGE];
				backLattice[idx].adj[DIR_RIGHT_FACE] += frontLattice[getIdx(left, y, z)].adj[DIR_RIGHT_FACE];
				backLattice[idx].adj[DIR_BACK_FACE] += frontLattice[getIdx(x, y, front)].adj[DIR_BACK_FACE];
				backLattice[idx].adj[DIR_TOP_FACE] += frontLattice[getIdx(x, bottom, z)].adj[DIR_TOP_FACE];
				backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] += frontLattice[getIdx(left, y, front)].adj[DIR_BACK_RIGHT_EDGE];
				backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] += frontLattice[getIdx(left, y, back)].adj[DIR_FRONT_RIGHT_EDGE];
				backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] += frontLattice[getIdx(left, bottom, z)].adj[DIR_TOP_RIGHT_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] += frontLattice[getIdx(left, top, z)].adj[DIR_BOTTOM_RIGHT_EDGE];
				backLattice[idx].adj[DIR_TOP_BACK_EDGE] += frontLattice[getIdx(x, bottom, front)].adj[DIR_TOP_BACK_EDGE];
				backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] += frontLattice[getIdx(x, top, front)].adj[DIR_BOTTOM_BACK_EDGE];

				for (int i = 0; i < 19; i++) {
					if (backLattice[idx].adj[i] < 0.0f) {
						backLattice[idx].adj[i] = 0.0f;
					} else if (backLattice[idx].adj[i] > 1.0f) {
						backLattice[idx].adj[i] = 1.0f;
					}
				}
			}
		}
	}

}

void LBM3D_1D_indices::collisionStep() {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;

	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int z = 0; z < GRID_DEPTH; z++) {

				int idx = getIdx(x, y, z);

				float macroDensity = calculateMacroscopicDensity(x, y, z);
				glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, z, macroDensity);

				velocities[x][y][z] = macroVelocity;

#ifdef DRAW_VELOCITY_ARROWS
				velocityArrows.push_back(glm::vec3(x, y, z));
				velocityArrows.push_back(glm::vec3(x, y, z) + velocities[x][y][z] * 2.0f);
#endif


				float leftTermMiddle = weightMiddle * macroDensity;
				float leftTermAxis = weightAxis * macroDensity;
				float leftTermNonaxial = weightNonaxial * macroDensity;

				float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
				float thirdTerm = 1.5f * macroVelocityDot;

				float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

				float dotProd = glm::dot(vRight, macroVelocity);
				float firstTerm = 3.0f * dotProd;
				float secondTerm = 4.5f * dotProd * dotProd;
				float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTop, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottom, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBackRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBackLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vFrontRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vFrontLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottomBack, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vBottomFront, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vTopLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				dotProd = glm::dot(vBottomRight, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

				dotProd = glm::dot(vBottomLeft, macroVelocity);
				firstTerm = 3.0f * dotProd;
				secondTerm = 4.5f * dotProd * dotProd;
				float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


				backLattice[idx].adj[DIR_MIDDLE] -= ITAU * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);
				backLattice[idx].adj[DIR_RIGHT_FACE] -= ITAU * (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
				backLattice[idx].adj[DIR_LEFT_FACE] -= ITAU * (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
				backLattice[idx].adj[DIR_BACK_FACE] -= ITAU * (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
				backLattice[idx].adj[DIR_FRONT_FACE] -= ITAU * (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
				backLattice[idx].adj[DIR_TOP_FACE] -= ITAU * (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
				backLattice[idx].adj[DIR_BOTTOM_FACE] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
				backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
				backLattice[idx].adj[DIR_BACK_LEFT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
				backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
				backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
				backLattice[idx].adj[DIR_TOP_BACK_EDGE] -= ITAU * (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
				backLattice[idx].adj[DIR_TOP_FRONT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
				backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
				backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
				backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
				backLattice[idx].adj[DIR_TOP_LEFT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
				backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
				backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] -= ITAU * (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


				for (int i = 0; i < 19; i++) {
					if (backLattice[idx].adj[i] < 0.0f) {
						backLattice[idx].adj[i] = 0.0f;
					} else if (backLattice[idx].adj[i] > 1.0f) {
						backLattice[idx].adj[i] = 1.0f;
					}
				}





			}
		}
	}

}

void LBM3D_1D_indices::moveParticles() {

	glm::vec3 adjVelocities[8];
	for (int i = 0; i < NUM_PARTICLES; i++) {
		float x = particleVertices[i].x;
		float y = particleVertices[i].y;
		float z = particleVertices[i].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[leftX][topY][backZ];
		adjVelocities[1] = velocities[rightX][topY][backZ];
		adjVelocities[2] = velocities[leftX][bottomY][backZ];
		adjVelocities[3] = velocities[rightX][bottomY][backZ];
		adjVelocities[4] = velocities[leftX][topY][frontZ];
		adjVelocities[5] = velocities[rightX][topY][frontZ];
		adjVelocities[6] = velocities[leftX][bottomY][frontZ];
		adjVelocities[7] = velocities[rightX][bottomY][frontZ];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;
		float depthRatio = z - backZ;

		glm::vec3 topBackVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec3 bottomBackVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec3 backVelocity = bottomBackVelocity * verticalRatio + topBackVelocity * (1.0f - verticalRatio);

		glm::vec3 topFrontVelocity = adjVelocities[4] * horizontalRatio + adjVelocities[5] * (1.0f - horizontalRatio);
		glm::vec3 bottomFrontVelocity = adjVelocities[6] * horizontalRatio + adjVelocities[7] * (1.0f - horizontalRatio);

		glm::vec3 frontVelocity = bottomFrontVelocity * verticalRatio + topFrontVelocity * (1.0f - verticalRatio);

		glm::vec3 finalVelocity = backVelocity * depthRatio + frontVelocity * (1.0f - depthRatio);

#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		particleArrows.push_back(particleVertices[i]);
#endif
		particleVertices[i] += finalVelocity;
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		glm::vec3 tmp = particleVertices[i] + 10.0f * finalVelocity;
		particleArrows.push_back(tmp);
#endif

		if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= GRID_WIDTH - 1 ||
			particleVertices[i].y <= 0.0f || particleVertices[i].y >= GRID_HEIGHT - 1 ||
			particleVertices[i].z <= 0.0f || particleVertices[i].z >= GRID_DEPTH - 1) {

			particleVertices[i] = glm::vec3(0.0f, respawnY, respawnZ++);
			if (respawnZ >= GRID_DEPTH - 1) {
				respawnZ = 0;
				respawnY++;
			}
			if (respawnY >= GRID_HEIGHT - 1) {
				respawnY = 0;
			}
		}

	}
}

void LBM3D_1D_indices::updateInlets() {

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;


	float macroDensity = 1.0f;
	glm::vec3 macroVelocity = glm::vec3(0.4f, 0.0f, 0.0f);


	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermNonaxial = weightNonaxial * macroDensity;

	float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	float dotProd = glm::dot(vRight, macroVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTop, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBackLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vFrontLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomBack, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomFront, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vBottomLeft, macroVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	for (int z = 0; z < GRID_DEPTH; z++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {

			int idx = getIdx(0, y, z);

			backLattice[idx].adj[DIR_MIDDLE] = middleEq;
			backLattice[idx].adj[DIR_RIGHT_FACE] = rightEq;
			backLattice[idx].adj[DIR_LEFT_FACE] = leftEq;
			backLattice[idx].adj[DIR_BACK_FACE] = backEq;
			backLattice[idx].adj[DIR_FRONT_FACE] = frontEq;
			backLattice[idx].adj[DIR_TOP_FACE] = topEq;
			backLattice[idx].adj[DIR_BOTTOM_FACE] = bottomEq;
			backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = backRightEq;
			backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = backLeftEq;
			backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = frontRightEq;
			backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = frontLeftEq;
			backLattice[idx].adj[DIR_TOP_BACK_EDGE] = topBackEq;
			backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = topFrontEq;
			backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = bottomBackEq;
			backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = bottomFrontEq;
			backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = topRightEq;
			backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = topLeftEq;
			backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = bottomRightEq;
			backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = bottomLeftEq;


			for (int i = 0; i < 19; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}
		}
	}






}

void LBM3D_1D_indices::updateColliders() {


	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int z = 0; z < GRID_DEPTH; z++) {

				if (/*z == 0 || z == GRID_DEPTH - 1 ||*/ sphereCollider[x][y][z]) {
					int idx = getIdx(x, y, z);

					float right = backLattice[idx].adj[DIR_RIGHT_FACE];
					float left = backLattice[idx].adj[DIR_LEFT_FACE];
					float back = backLattice[idx].adj[DIR_BACK_FACE];
					float front = backLattice[idx].adj[DIR_FRONT_FACE];
					float top = backLattice[idx].adj[DIR_TOP_FACE];
					float bottom = backLattice[idx].adj[DIR_BOTTOM_FACE];
					float backRight = backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
					float backLeft = backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
					float frontRight = backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
					float frontLeft = backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
					float topBack = backLattice[idx].adj[DIR_TOP_BACK_EDGE];
					float topFront = backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
					float bottomBack = backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
					float bottomFront = backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
					float topRight = backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
					float topLeft = backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
					float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
					float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];


					backLattice[idx].adj[DIR_RIGHT_FACE] = left;
					backLattice[idx].adj[DIR_LEFT_FACE] = right;
					backLattice[idx].adj[DIR_BACK_FACE] = front;
					backLattice[idx].adj[DIR_FRONT_FACE] = back;
					backLattice[idx].adj[DIR_TOP_FACE] = bottom;
					backLattice[idx].adj[DIR_BOTTOM_FACE] = top;
					backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] = frontLeft;
					backLattice[idx].adj[DIR_BACK_LEFT_EDGE] = frontRight;
					backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] = backLeft;
					backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] = backRight;
					backLattice[idx].adj[DIR_TOP_BACK_EDGE] = bottomFront;
					backLattice[idx].adj[DIR_TOP_FRONT_EDGE] = bottomBack;
					backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] = topFront;
					backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] = topBack;
					backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] = bottomLeft;
					backLattice[idx].adj[DIR_TOP_LEFT_EDGE] = bottomRight;
					backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] = topLeft;
					backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] = topRight;

					float macroDensity = calculateMacroscopicDensity(x, y, z);
					glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, z, macroDensity);
					velocities[x][y][z] = macroVelocity;

				}



			}
		}


	}



}

void LBM3D_1D_indices::initBuffers() {

	glGenVertexArrays(1, &colliderVAO);
	glBindVertexArray(colliderVAO);
	glGenBuffers(1, &colliderVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colliderVBO);	

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * colliderVertices.size(), &colliderVertices[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);


#ifdef DRAW_VELOCITY_ARROWS
	// Velocity arrows
	glGenVertexArrays(1, &velocityVAO);
	glBindVertexArray(velocityVAO);
	glGenBuffers(1, &velocityVBO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	// Particle arrows
	glGenVertexArrays(1, &particleArrowsVAO);
	glBindVertexArray(particleArrowsVAO);
	glGenBuffers(1, &particleArrowsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
#endif


	glBindVertexArray(0);


}

void LBM3D_1D_indices::initLattice() {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;
	for (int x = 0; x < GRID_WIDTH; x++) {
		for (int y = 0; y < GRID_HEIGHT; y++) {
			for (int z = 0; z < GRID_DEPTH; z++) {
				int idx = getIdx(x, y, z);
				frontLattice[idx].adj[DIR_MIDDLE] = weightMiddle;
				for (int i = 1; i <= 6; i++) {
					frontLattice[idx].adj[i] = weightAxis;
				}
				for (int i = 7; i <= 18; i++) {
					frontLattice[idx].adj[i] = weightNonaxial;
				}
			}
		}
	}


}

void LBM3D_1D_indices::initColliders() {

	// test sphere collider
	//glm::vec3 center(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, GRID_DEPTH / 2.0f);
	//float radius = GRID_DEPTH / 2.0f;

	//for (int x = 0; x < GRID_WIDTH; x++) {
	//	for (int y = 0; y < GRID_HEIGHT; y++) {
	//		for (int z = 0; z < GRID_DEPTH; z++) {

	//			if (glm::distance(center, glm::vec3(x, y, z)) <= radius) {
	//				sphereCollider[x][y][z] = true;
	//			}

	//		}
	//	}
	//}


	for (int x = GRID_WIDTH / 3.0f; x < GRID_WIDTH / 2.0f; x++) {
		for (int y = GRID_HEIGHT / 4.0f; y < GRID_HEIGHT / 3.0f; y++) {
			for (int z = GRID_DEPTH / 3.0f; z < GRID_DEPTH / 2.0f; z++) {
				sphereCollider[x][y][z] = true;
				colliderVertices.push_back(glm::vec3(x, y, z));
			}
		}
	}




}

void LBM3D_1D_indices::swapLattices() {
	Node *tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;
}

float LBM3D_1D_indices::calculateMacroscopicDensity(int x, int y, int z) {
	float macroDensity = 0.0f;
	int idx = getIdx(x, y, z);
	for (int i = 0; i < 19; i++) {
		macroDensity += backLattice[idx].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM3D_1D_indices::calculateMacroscopicVelocity(int x, int y, int z, float macroDensity) {

	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	int idx = getIdx(x, y, z);
	//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
	macroVelocity += vLeft * backLattice[idx].adj[DIR_LEFT_FACE];
	macroVelocity += vFront * backLattice[idx].adj[DIR_FRONT_FACE];
	macroVelocity += vBottom * backLattice[idx].adj[DIR_BOTTOM_FACE];
	macroVelocity += vFrontLeft * backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
	macroVelocity += vBackLeft * backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
	macroVelocity += vBottomLeft * backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];
	macroVelocity += vTopLeft * backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
	macroVelocity += vBottomFront * backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
	macroVelocity += vTopFront * backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
	macroVelocity += vRight * backLattice[idx].adj[DIR_RIGHT_FACE];
	macroVelocity += vBack * backLattice[idx].adj[DIR_BACK_FACE];
	macroVelocity += vTop * backLattice[idx].adj[DIR_TOP_FACE];
	macroVelocity += vBackRight * backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
	macroVelocity += vFrontRight * backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
	macroVelocity += vTopRight * backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
	macroVelocity += vBottomRight * backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
	macroVelocity += vTopBack * backLattice[idx].adj[DIR_TOP_BACK_EDGE];
	macroVelocity += vBottomBack * backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
	macroVelocity /= macroDensity;

	return macroVelocity;
}
