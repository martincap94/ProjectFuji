#include "LBM2D_1D_indices.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <glm\gtx\norm.hpp>
#include "CUDAUtils.cuh"

#include <omp.h>


__constant__ int d_latticeWidth;		///< Lattice width constant on the device
__constant__ int d_latticeHeight;		///< Lattice height constant on the device
__constant__ int d_latticeSize;			///< Lattice size constant on the device (latticeWidth * latticeHeight)
__constant__ float d_tau;				///< Tau value on the device
__constant__ float d_itau;				///< Inverse tau value (1.0f / tau) on the device
__constant__ int d_mirrorSides;			///< Whether to mirror sides (cycle) on the device
//__constant__ int d_visualizeVelocity;

__device__ int d_respawnIndex = 0;		///< Respawn index (y coordinate) for particle respawn, not used
__constant__ int d_respawnMinY;			///< Minimum y respawn coordinate, not used
__constant__ int d_respawnMaxY;			///< Maximum y respawn coordinate, not used

__constant__ glm::vec3 d_directionVectors[NUM_2D_DIRECTIONS];	///< Constant array of direction vectors


/// Returns uniform random between 0.0 and 1.0. Provided from different student's work.
__device__ __host__ float rand2D(int x, int y) {
	int n = x + y * 57;
	n = (n << 13) ^ n;

	return ((1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f) + 1.0f) * 0.5f;
}

/// Returns the flattened index using the device constants and provided coordinates.
__device__ int getIdxKernel(int x, int y) {
	return x + y * d_latticeWidth;
}


/// Maps the value to the viridis color map.
__device__ glm::vec3 mapToViridis2D(float val) {
	val = glm::clamp(val, 0.0f, 1.0f);
	int discreteVal = (int)(val * 255.0f);
	return glm::vec3(viridis_cm[discreteVal][0], viridis_cm[discreteVal][1], viridis_cm[discreteVal][2]);
}


/// Kernel for moving particles that uses OpenGL interoperability.
/**
	Kernel for moving particles that uses OpenGL interoperability for setting particle positions and colors.
	If the particles venture beyond the simulation bounding volume, they are randomly respawned.
	If we use side mirroring (cycling), particles that go beyond side walls (on the y axis) will be mirrored/cycled to the other side of the bounding volume.
	\param[in] particleVertices		Vertices (positions stored in VBO) of particles to be updated/moved.
	\param[in] velocities			Array of velocities that will act on the particles.
	\param[in] numParticles			Number of particles.
	\param[in] particleColors		VBO of particle colors.
*/
__global__ void moveParticlesKernelInterop(glm::vec3 *particleVertices, glm::vec2 *velocities, int *numParticles, glm::vec3 *particleColors) {


	glm::vec2 adjVelocities[4];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	while (idx < *numParticles) {
		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;


		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[getIdxKernel(leftX, topY)];
		adjVelocities[1] = velocities[getIdxKernel(rightX, topY)];
		adjVelocities[2] = velocities[getIdxKernel(leftX, bottomY)];
		adjVelocities[3] = velocities[getIdxKernel(rightX, bottomY)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);



		//particleVertices[idx] += make_float3(finalVelocity.x, 0.0f);
		particleVertices[idx].x += finalVelocity.x;
		particleVertices[idx].y += finalVelocity.y;


		//particleColors[idx] = glm::vec3(glm::length2(finalVelocity) * 4.0f);
		//particleColors[idx] = mapToColor(glm::length2(finalVelocity) * 4.0f);
		particleColors[idx] = mapToViridis2D(glm::length2(finalVelocity) * 4.0f);

		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
			particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1) {
			if (d_mirrorSides) {
				if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1) {
					particleVertices[idx].x = 0.0f;
					particleVertices[idx].y = rand2D(idx, y) * (d_latticeHeight - 1);

					////particleVertices[idx].y = d_respawnIndex++;
					//particleVertices[idx].y = d_respawnIndex;
					//atomicAdd(&d_respawnIndex, 1);
					//if (d_respawnIndex >= d_respawnMaxY) {
					//	//d_respawnIndex = d_respawnMinY;
					//	atomicExch(&d_respawnIndex, d_respawnMinY);
					//}
				} else {
					particleVertices[idx].y = (float)((int)(particleVertices[idx].y + d_latticeHeight - 1) % (d_latticeHeight - 1));
				}
			} else {
				particleVertices[idx].x = 0.0f;
				particleVertices[idx].y = rand2D(idx, y) * (d_latticeHeight - 1);

				////particleVertices[idx].y = d_respawnIndex++;
				//particleVertices[idx].y = d_respawnIndex;
				//atomicAdd(&d_respawnIndex, 1);
				//if (d_respawnIndex >= d_respawnMaxY) {
				//	//d_respawnIndex = d_respawnMinY;
				//	atomicExch(&d_respawnIndex, d_respawnMinY);
				//}
			}
			particleVertices[idx].z = 0.0f;
		}

		idx += blockDim.x * gridDim.x;


	}
}


/// Kernel for clearing the back lattice.
/**
	Kernel that clears the back lattice.
	\param[in] backLattice	Pointer to the back lattice to be cleared.
*/
__global__ void clearBackLatticeKernel(Node *backLattice) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < d_latticeSize) {
		for (int i = 0; i < 9; i++) {
			backLattice[idx].adj[i] = 0.0f;
		}
	}
}


/// Kernel that streams the microscopic particles from the previous frame.
/**
	Kernel that streams the microscopic particles from the previous frame.
	\param[in] backLatice		Lattice that will be used in the current frame (the one we are currently updating).
	\param[in] frontLattice	Lattice from the previous frame from which we stream the particles.
*/
__global__ void streamingStepKernel(Node *backLattice, Node *frontLattice) {

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < d_latticeSize) {

		int x = idx % d_latticeWidth;
		int y = (idx / d_latticeWidth) % d_latticeHeight;

		backLattice[idx].adj[DIR_MIDDLE] += frontLattice[idx].adj[DIR_MIDDLE];

		int right;
		int left;
		int top;
		int bottom;

		right = x + 1;
		left = x - 1;
		top = y + 1;
		bottom = y - 1;
		if (right > d_latticeWidth - 1) {
			right = d_latticeWidth - 1;
		}
		if (left < 0) {
			left = 0;
		}
		if (top > d_latticeHeight - 1) {
			top = d_latticeHeight - 1;
		}
		if (bottom < 0) {
			bottom = 0;
		}


		backLattice[idx].adj[DIR_RIGHT] += frontLattice[getIdxKernel(left, y)].adj[DIR_RIGHT];
		backLattice[idx].adj[DIR_TOP] += frontLattice[getIdxKernel(x, bottom)].adj[DIR_TOP];
		backLattice[idx].adj[DIR_LEFT] += frontLattice[getIdxKernel(right, y)].adj[DIR_LEFT];
		backLattice[idx].adj[DIR_BOTTOM] += frontLattice[getIdxKernel(x, top)].adj[DIR_BOTTOM];
		backLattice[idx].adj[DIR_TOP_RIGHT] += frontLattice[getIdxKernel(left, bottom)].adj[DIR_TOP_RIGHT];
		backLattice[idx].adj[DIR_TOP_LEFT] += frontLattice[getIdxKernel(right, bottom)].adj[DIR_TOP_LEFT];
		backLattice[idx].adj[DIR_BOTTOM_LEFT] += frontLattice[getIdxKernel(right, top)].adj[DIR_BOTTOM_LEFT];
		backLattice[idx].adj[DIR_BOTTOM_RIGHT] += frontLattice[getIdxKernel(left, top)].adj[DIR_BOTTOM_RIGHT];

		for (int i = 0; i < 9; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}


}


/// Kernel for updating the inlets.
/**
	Kernel for updating the inlets. Acts the same way as collision step but with predetermined velocity and density.
	The inlet is the left wall of the simulation bounding volume.
	\param[in] backLattice		The back lattice where we update node values.
	\param[in] velocities		Velocities array for the lattice.
	\param[in] inletVelocity	Our desired inlet velocity.
*/
__global__ void updateInletsKernel(Node *lattice, glm::vec3 inletVelocity) {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	//glm::vec3 macroVelocity = inletVelocity; // unnecessary variable -> remove

	const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
	const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
	const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
	const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
	const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
	const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
	const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
	const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

	// let's find the equilibrium
	float leftTermMiddle = weightMiddle * macroDensity;
	float leftTermAxis = weightAxis * macroDensity;
	float leftTermDiagonal = weightDiagonal * macroDensity;

	// optimize these operations later

	float macroVelocityDot = glm::dot(inletVelocity, inletVelocity);
	float thirdTerm = 1.5f * macroVelocityDot / LAT_SPEED_SQ;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	// this can all be rewritten into arrays + for cycles!
	float dotProd = glm::dot(vRight, inletVelocity);
	float firstTerm = 3.0f * dotProd / LAT_SPEED;
	float secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vTop, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(vLeft, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottom, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopRight, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vTopLeft, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomLeft, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(vBottomRight, inletVelocity);
	firstTerm = 3.0f * dotProd / LAT_SPEED;
	secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
	float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int x = idx % d_latticeWidth;

	if (x == 0 && idx < d_latticeSize) {

		lattice[idx].adj[DIR_MIDDLE] = middleEq;
		lattice[idx].adj[DIR_RIGHT] = rightEq;
		lattice[idx].adj[DIR_TOP] = topEq;
		lattice[idx].adj[DIR_LEFT] = leftEq;
		lattice[idx].adj[DIR_BOTTOM] = bottomEq;
		lattice[idx].adj[DIR_TOP_RIGHT] = topRightEq;
		lattice[idx].adj[DIR_TOP_LEFT] = topLeftEq;
		lattice[idx].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		lattice[idx].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (lattice[idx].adj[i] < 0.0f) {
				lattice[idx].adj[i] = 0.0f;
			} else if (lattice[idx].adj[i] > 1.0f) {
				lattice[idx].adj[i] = 1.0f;
			}
		}
	}

}

/// Kernel for updating colliders/obstacles in the lattice.
/**
	Updates colliders/obstacles by using the full bounce back approach.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
	\param[in] heightMap		Height map of the scene.
*/
__global__ void updateCollidersKernel(Node *backLattice, bool *tCol) {


	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < d_latticeSize) {
		if (tCol[idx]) {

			float right = backLattice[idx].adj[DIR_RIGHT];
			float top = backLattice[idx].adj[DIR_TOP];
			float left = backLattice[idx].adj[DIR_LEFT];
			float bottom = backLattice[idx].adj[DIR_BOTTOM];
			float topRight = backLattice[idx].adj[DIR_TOP_RIGHT];
			float topLeft = backLattice[idx].adj[DIR_TOP_LEFT];
			float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT];
			float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT];
			backLattice[idx].adj[DIR_RIGHT] = left;
			backLattice[idx].adj[DIR_TOP] = bottom;
			backLattice[idx].adj[DIR_LEFT] = right;
			backLattice[idx].adj[DIR_BOTTOM] = top;
			backLattice[idx].adj[DIR_TOP_RIGHT] = bottomLeft;
			backLattice[idx].adj[DIR_TOP_LEFT] = bottomRight;
			backLattice[idx].adj[DIR_BOTTOM_LEFT] = topRight;
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] = topLeft;
		}
	}
}


/// Kernel for calculating the collision operator.
/**
	Kernel that calculates the collision operator using Bhatnagar-Gross-Krook operator.
	Uses shared memory for speedup.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
*/
__global__ void collisionStepKernel(Node *backLattice, glm::vec2 *velocities) {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1D array kernel
	int cacheIdx = threadIdx.x;

	extern __shared__ Node cache[];


	if (idx < d_latticeSize) {

		cache[cacheIdx] = backLattice[idx];


		float macroDensity = 0.0f;
		for (int i = 0; i < 9; i++) {
			macroDensity += cache[cacheIdx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		macroVelocity += LAT_SPEED * d_directionVectors[DIR_RIGHT] * cache[cacheIdx].adj[DIR_RIGHT];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_TOP] * cache[cacheIdx].adj[DIR_TOP];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_LEFT] * cache[cacheIdx].adj[DIR_LEFT];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_BOTTOM] * cache[cacheIdx].adj[DIR_BOTTOM];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_TOP_RIGHT] * cache[cacheIdx].adj[DIR_TOP_RIGHT];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_TOP_LEFT] * cache[cacheIdx].adj[DIR_TOP_LEFT];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_BOTTOM_LEFT] * cache[cacheIdx].adj[DIR_BOTTOM_LEFT];
		macroVelocity += LAT_SPEED * d_directionVectors[DIR_BOTTOM_RIGHT] * cache[cacheIdx].adj[DIR_BOTTOM_RIGHT];
		macroVelocity /= macroDensity;


		//velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);
		velocities[idx].x = macroVelocity.x;
		velocities[idx].y = macroVelocity.y;


		// let's find the equilibrium
		float leftTermMiddle = weightMiddle * macroDensity;
		float leftTermAxis = weightAxis * macroDensity;
		float leftTermDiagonal = weightDiagonal * macroDensity;

		// optimize these operations later

		float thirdTerm = 1.5f * glm::dot(macroVelocity, macroVelocity) / LAT_SPEED_SQ;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		// this can all be rewritten into arrays + for cycles!
		float dotProd = glm::dot(d_directionVectors[DIR_RIGHT], macroVelocity);
		float firstTerm = 3.0f * dotProd / LAT_SPEED;
		float secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(d_directionVectors[DIR_TOP], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(d_directionVectors[DIR_LEFT], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(d_directionVectors[DIR_BOTTOM], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(d_directionVectors[DIR_TOP_RIGHT], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(d_directionVectors[DIR_TOP_LEFT], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float topLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(d_directionVectors[DIR_BOTTOM_LEFT], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomLeftEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(d_directionVectors[DIR_BOTTOM_RIGHT], macroVelocity);
		firstTerm = 3.0f * dotProd / LAT_SPEED;
		secondTerm = 4.5f * dotProd * dotProd / LAT_SPEED_SQ;
		float bottomRightEq = leftTermDiagonal + leftTermDiagonal * (firstTerm + secondTerm - thirdTerm);

		cache[cacheIdx].adj[DIR_MIDDLE] -= d_itau * (cache[cacheIdx].adj[DIR_MIDDLE] - middleEq);
		cache[cacheIdx].adj[DIR_RIGHT] -= d_itau * (cache[cacheIdx].adj[DIR_RIGHT] - rightEq);
		cache[cacheIdx].adj[DIR_TOP] -= d_itau * (cache[cacheIdx].adj[DIR_TOP] - topEq);
		cache[cacheIdx].adj[DIR_LEFT] -= d_itau * (cache[cacheIdx].adj[DIR_LEFT] - leftEq);
		cache[cacheIdx].adj[DIR_BOTTOM] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM] - bottomEq);
		cache[cacheIdx].adj[DIR_TOP_RIGHT] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_RIGHT] - topRightEq);
		cache[cacheIdx].adj[DIR_TOP_LEFT] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_LEFT] - topLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_LEFT] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_RIGHT] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


		for (int i = 0; i < 9; i++) {
			if (cache[cacheIdx].adj[i] < 0.0f) {
				cache[cacheIdx].adj[i] = 0.0f;
			} else if (cache[cacheIdx].adj[i] > 1.0f) {
				cache[cacheIdx].adj[i] = 1.0f;
			}
		}

		backLattice[idx] = cache[cacheIdx];

	}
}







LBM2D_1D_indices::LBM2D_1D_indices() {
}

LBM2D_1D_indices::LBM2D_1D_indices(glm::ivec3 dim, string sceneFilename, float tau, ParticleSystemLBM *particleSystem, int numThreads) : LBM(nullptr, dim, sceneFilename, tau, particleSystem), numThreads(numThreads) {
	

	initScene();

	frontLattice = new Node[latticeSize]();
	backLattice = new Node[latticeSize]();
	velocities = new glm::vec2[latticeSize]();

	cudaMalloc((void**)&d_frontLattice, sizeof(Node) * latticeSize);
	cudaMalloc((void**)&d_backLattice, sizeof(Node) * latticeSize);
	cudaMalloc((void**)&d_velocities, sizeof(glm::vec2) * latticeSize);

	cudaMemcpyToSymbol(d_latticeWidth, &latticeWidth, sizeof(int));
	cudaMemcpyToSymbol(d_latticeHeight, &latticeHeight, sizeof(int));
	cudaMemcpyToSymbol(d_latticeSize, &latticeSize, sizeof(int));
	cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_itau, &itau, sizeof(float));
	cudaMemcpyToSymbol(d_mirrorSides, &mirrorSides, sizeof(int));
	cudaMemcpyToSymbol(d_directionVectors, &directionVectors, sizeof(glm::vec3) * NUM_2D_DIRECTIONS);


	cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particleSystem->vbo, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cudaParticleColorsVBO, particleSystem->colorsVBO, cudaGraphicsMapFlagsWriteDiscard);



	initBuffers();
	initLattice();
	//updateInlets(frontLattice);

	cudaMemcpy(d_backLattice, backLattice, sizeof(Node) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec2) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node) * latticeSize, cudaMemcpyHostToDevice);

	numBlocks = (int)ceil(latticeSize / this->numThreads) + 1;


}

void LBM2D_1D_indices::resetSimulation() {
	cout << "Resetting simulation..." << endl;
	particleSystem->initParticlePositions(latticeWidth, latticeHeight, tCol->area);
	for (int i = 0; i < latticeWidth * latticeHeight; i++) {
		for (int j = 0; j < 9; j++) {
			backLattice[i].adj[j] = 0.0f;
		}
		velocities[i] = glm::vec3(0.0f);
	}
	initLattice();

	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node) * latticeWidth * latticeHeight, cudaMemcpyHostToDevice);
	cudaMemcpy(d_backLattice, backLattice, sizeof(Node) * latticeWidth * latticeHeight, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec2) * latticeWidth * latticeHeight, cudaMemcpyHostToDevice);

}


void LBM2D_1D_indices::switchToCPU() {
	cout << "Copying data back to CPU for simulation..." << endl;
	cudaMemcpy(frontLattice, d_frontLattice, sizeof(Node) * latticeSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(backLattice, d_backLattice, sizeof(Node) * latticeSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(velocities, d_velocities, sizeof(glm::vec2) * latticeSize, cudaMemcpyDeviceToHost);

	particleSystem->copyDataFromVBOtoCPU();
}

void LBM2D_1D_indices::synchronize() {
	cudaDeviceSynchronize();
}



LBM2D_1D_indices::~LBM2D_1D_indices() {
	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;

	delete tCol;

	cudaFree(d_frontLattice);
	cudaFree(d_backLattice);
	cudaFree(d_tCol);
	cudaFree(d_velocities);

	cudaGraphicsUnregisterResource(cudaParticleVerticesVBO);
	cudaGraphicsUnregisterResource(cudaParticleColorsVBO);

}

void LBM2D_1D_indices::recalculateVariables() {
	LBM::recalculateVariables();
	cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_itau, &itau, sizeof(float));
}

void LBM2D_1D_indices::initScene() {
	tCol = new LatticeCollider(sceneFilename);

	latticeWidth = tCol->width;
	latticeHeight = tCol->height;
	latticeDepth = 1;
	latticeSize = latticeWidth * latticeHeight;

	precomputeRespawnRange();

	cudaMalloc((void**)&d_tCol, sizeof(bool) * latticeSize);
	cudaMemcpy(d_tCol, &tCol->area[0], sizeof(bool) * latticeSize, cudaMemcpyHostToDevice);


	particleVertices = particleSystem->particleVertices;
	d_numParticles = particleSystem->d_numParticles;

	particleSystem->initParticlePositions(latticeWidth, latticeHeight, tCol->area);

}

void LBM2D_1D_indices::draw(ShaderProgram &shader) {
	//glPointSize(0.4f);
	//shader.setVec3("u_Color", glm::vec3(0.4f, 0.4f, 0.1f));
	//glUseProgram(shader.id);

	//glBindVertexArray(vao);
	//glDrawArrays(GL_POINTS, 0, latticeWidth * latticeHeight);


	//cout << "Velocity arrows size = " << velocityArrows.size() << endl;

#ifdef DRAW_VELOCITY_ARROWS
	shader.setVec3("u_Color", glm::vec3(0.2f, 0.3f, 1.0f));
	glBindVertexArray(velocityVAO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * velocityArrows.size(), &velocityArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, velocityArrows.size());
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	shader.setVec3("u_Color", glm::vec3(0.8f, 1.0f, 0.6f));

	glBindVertexArray(particleArrowsVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particleArrows.size(), &particleArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, particleArrows.size());
#endif

	// Draw scene collider
	tCol->draw(shader);





}

void LBM2D_1D_indices::doStep() {

	clearBackLattice();

	updateInlets();
	streamingStep();
	updateColliders();

	collisionStep();

	//collisionStepStreamlined();

	moveParticles();

	swapLattices();


}

void LBM2D_1D_indices::doStepCUDA() {

	// ============================================= clear back lattice CUDA
	clearBackLatticeKernel << <numBlocks, numThreads >> > (d_backLattice);

	// ============================================= update inlets CUDA
	updateInletsKernel << <numBlocks, numThreads >> > (d_backLattice, inletVelocity);

	// ============================================= streaming step CUDA
	streamingStepKernel << <numBlocks, numThreads >> > (d_backLattice, d_frontLattice);

	// ============================================= update colliders CUDA
	updateCollidersKernel << <numBlocks, numThreads >> > (d_backLattice, d_tCol);

	// ============================================= collision step CUDA
	collisionStepKernel << <numBlocks, numThreads, numThreads * sizeof(Node) >> > (d_backLattice, d_velocities);

	// ============================================= move particles CUDA - different respawn from CPU !!!

	glm::vec3 *dptr;
	cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cudaParticleVerticesVBO);
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	glm::vec3 *d_particleColors;
	cudaGraphicsMapResources(1, &cudaParticleColorsVBO, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_particleColors, &num_bytes, cudaParticleColorsVBO);


	moveParticlesKernelInterop << <numBlocks, numThreads >> > (dptr, d_velocities, d_numParticles, d_particleColors);

	cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0);
	cudaGraphicsUnmapResources(1, &cudaParticleColorsVBO, 0);

	swapLattices();
}

void LBM2D_1D_indices::clearBackLattice() {
	for (int i = 0; i < latticeSize; i++) {
		for (int j = 0; j < 9; j++) {
			backLattice[i].adj[j] = 0.0f;
		}
	}
#ifdef DRAW_VELOCITY_ARROWS
	velocityArrows.clear();
#endif
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	particleArrows.clear();
#endif
}

void LBM2D_1D_indices::streamingStep() {

	for (int x = 0; x < latticeWidth; x++) {
//#pragma omp parallel for/* simd */
		for (int y = 0; y < latticeHeight; y++) {

			backLattice[getIdx(x, y)].adj[DIR_MIDDLE] += frontLattice[getIdx(x, y)].adj[DIR_MIDDLE];

			int right;
			int left;
			int top;
			int bottom;

			right = x + 1;
			left = x - 1;
			top = y + 1;
			bottom = y - 1;
			if (right > latticeWidth - 1) {
				right = latticeWidth - 1;
			}
			if (left < 0) {
				left = 0;
			}
			if (top > latticeHeight - 1) {
				top = latticeHeight - 1;
			}
			if (bottom < 0) {
				bottom = 0;
			}


			backLattice[getIdx(x, y)].adj[DIR_RIGHT] += frontLattice[getIdx(left, y)].adj[DIR_RIGHT];
			backLattice[getIdx(x, y)].adj[DIR_TOP] += frontLattice[getIdx(x, bottom)].adj[DIR_TOP];
			backLattice[getIdx(x, y)].adj[DIR_LEFT] += frontLattice[getIdx(right, y)].adj[DIR_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM] += frontLattice[getIdx(x, top)].adj[DIR_BOTTOM];
			backLattice[getIdx(x, y)].adj[DIR_TOP_RIGHT] += frontLattice[getIdx(left, bottom)].adj[DIR_TOP_RIGHT];
			backLattice[getIdx(x, y)].adj[DIR_TOP_LEFT] += frontLattice[getIdx(right, bottom)].adj[DIR_TOP_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM_LEFT] += frontLattice[getIdx(right, top)].adj[DIR_BOTTOM_LEFT];
			backLattice[getIdx(x, y)].adj[DIR_BOTTOM_RIGHT] += frontLattice[getIdx(left, top)].adj[DIR_BOTTOM_RIGHT];

			for (int i = 0; i < 9; i++) {
				if (backLattice[getIdx(x, y)].adj[i] < 0.0f) {
					backLattice[getIdx(x, y)].adj[i] = 0.0f;
				} else if (backLattice[getIdx(x, y)].adj[i] > 1.0f) {
					backLattice[getIdx(x, y)].adj[i] = 1.0f;
				}
			}

		}
	}
}

void LBM2D_1D_indices::collisionStep() {

	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < latticeWidth; x++) {
//#pragma omp parallel for /*simd*/
		for (int y = 0; y < latticeHeight; y++) {


			float macroDensity = calculateMacroscopicDensity(x, y);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);

			int idx = getIdx(x, y);
			velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);

#ifdef DRAW_VELOCITY_ARROWS
			velocityArrows.push_back(glm::vec3(x, y, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[idx] * 5.0f, -1.0f) + glm::vec3(x, y, 0.0f));
#endif


			// let's find the equilibrium
			float leftTermMiddle = weightMiddle * macroDensity;
			float leftTermAxis = weightAxis * macroDensity;
			float leftTermDiagonal = weightDiagonal * macroDensity;

			// optimize these operations later

			float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
			float thirdTerm = 1.5f * macroVelocityDot;

			float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

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

#ifdef SUBGRID_EXPERIMENTAL
			// SUBGRID MODEL

			float middleTensor;
			float rightTensor;
			float topTensor;
			float leftTensor;
			float bottomTensor;
			float topRightTensor;
			float topLeftTensor;
			float bottomLeftTensor;
			float bottomRightTensor;

			float pi[9];

			/*float sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[i], directionVectors[i]);
			}*/

			float sum = 0.0f;
			middleTensor = sum * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[1], directionVectors[1]);
			}
			rightTensor = sum * (backLattice[idx].adj[DIR_RIGHT] - rightEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[2], directionVectors[2]);
			}
			topTensor = sum * (backLattice[idx].adj[DIR_TOP] - topEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[3], directionVectors[3]);
			}
			leftTensor = sum * (backLattice[idx].adj[DIR_LEFT] - leftEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[4], directionVectors[4]);
			}
			bottomTensor = sum * (backLattice[idx].adj[DIR_BOTTOM] - bottomEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[5], directionVectors[5]);
			}
			topRightTensor = sum * (backLattice[idx].adj[DIR_TOP_RIGHT] - topRightEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[6], directionVectors[6]);
			}
			topLeftTensor = sum * (backLattice[idx].adj[DIR_TOP_LEFT] - topLeftEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[7], directionVectors[7]);
			}
			bottomLeftTensor = sum * (backLattice[idx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);

			sum = 0.0f;
			for (int i = 0; i < 9; i++) {
				sum += glm::dot(directionVectors[8], directionVectors[8]);
			}
			bottomRightTensor = sum * (backLattice[idx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);



			sum = 0.0f;
			sum += middleTensor * middleTensor;
			sum += rightTensor * rightTensor;
			sum += topTensor * topTensor;
			sum += leftTensor * leftTensor;
			sum += bottomTensor * bottomTensor;
			sum += topRightTensor * topRightTensor;
			sum += topLeftTensor * topLeftTensor;
			sum += bottomLeftTensor * bottomLeftTensor;
			sum += bottomRightTensor * bottomRightTensor;

			float S = (-nu + sqrtf(nu * nu + 18.0f * SMAG_C * sqrtf(sum))) / (6.0f * SMAG_C * SMAG_C);

			tau = 3.0f * (nu + SMAG_C * SMAG_C * S) + 0.5f;
			itau = 1.0f / tau;
			//cout << "TAU = " << tau << endl;
#endif

			backLattice[idx].adj[DIR_MIDDLE] -= itau * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);
			backLattice[idx].adj[DIR_RIGHT] -= itau * (backLattice[idx].adj[DIR_RIGHT] - rightEq);
			backLattice[idx].adj[DIR_TOP] -= itau * (backLattice[idx].adj[DIR_TOP] - topEq);
			backLattice[idx].adj[DIR_LEFT] -= itau * (backLattice[idx].adj[DIR_LEFT] - leftEq);
			backLattice[idx].adj[DIR_BOTTOM] -= itau * (backLattice[idx].adj[DIR_BOTTOM] - bottomEq);
			backLattice[idx].adj[DIR_TOP_RIGHT] -= itau * (backLattice[idx].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[idx].adj[DIR_TOP_LEFT] -= itau * (backLattice[idx].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_LEFT] -= itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] -= itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}

		}
	}
}

void LBM2D_1D_indices::collisionStepStreamlined() {


	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {

			float macroDensity = calculateMacroscopicDensity(x, y);

			glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, macroDensity);

			int idx = getIdx(x, y);
			velocities[idx] = glm::vec2(macroVelocity.x, macroVelocity.y);

#ifdef DRAW_VELOCITY_ARROWS
			velocityArrows.push_back(glm::vec3(x, y, -0.5f));
			velocityArrows.push_back(glm::vec3(velocities[idx] * 5.0f, -1.0f) + glm::vec3(x, y, 0.0f));
#endif

			// let's find the equilibrium
			float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
			float leftTermAxis = WEIGHT_AXIS * macroDensity;
			float leftTermDiagonal = WEIGHT_DIAGONAL * macroDensity;

			float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
			float thirdTerm = 1.5f * macroVelocityDot;

			float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

			float rightEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);
			float topEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);
			float leftEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);
			float bottomEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);
			float topRightEq = leftTermDiagonal * (1.0f + 3.0f * (macroVelocity.x + macroVelocity.y) +
												   4.5f * (macroVelocity.x + macroVelocity.y) * (macroVelocity.x + macroVelocity.y) - thirdTerm);
			float topLeftEq = leftTermDiagonal * (1.0f + 3.0f * (-macroVelocity.x + macroVelocity.y) +
												  4.5f * (-macroVelocity.x + macroVelocity.y) * (-macroVelocity.x + macroVelocity.y) - thirdTerm);
			float bottomLeftEq = leftTermDiagonal * (1.0f + 3.0f * (-macroVelocity.x - macroVelocity.y) +
													 4.5f * (-macroVelocity.x - macroVelocity.y) * (-macroVelocity.x - macroVelocity.y) - thirdTerm);
			float bottomRightEq = leftTermDiagonal * (1.0f + 3.0f * (macroVelocity.x - macroVelocity.y) +
													  4.5f * (macroVelocity.x - macroVelocity.y) * (macroVelocity.x - macroVelocity.y) - thirdTerm);

			backLattice[idx].adj[DIR_MIDDLE] -= itau * (backLattice[idx].adj[DIR_MIDDLE] - middleEq);
			backLattice[idx].adj[DIR_RIGHT] -= itau * (backLattice[idx].adj[DIR_RIGHT] - rightEq);
			backLattice[idx].adj[DIR_TOP] -= itau * (backLattice[idx].adj[DIR_TOP] - topEq);
			backLattice[idx].adj[DIR_LEFT] -= itau * (backLattice[idx].adj[DIR_LEFT] - leftEq);
			backLattice[idx].adj[DIR_BOTTOM] -= itau * (backLattice[idx].adj[DIR_BOTTOM] - bottomEq);
			backLattice[idx].adj[DIR_TOP_RIGHT] -= itau * (backLattice[idx].adj[DIR_TOP_RIGHT] - topRightEq);
			backLattice[idx].adj[DIR_TOP_LEFT] -= itau * (backLattice[idx].adj[DIR_TOP_LEFT] - topLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_LEFT] -= itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT] - bottomLeftEq);
			backLattice[idx].adj[DIR_BOTTOM_RIGHT] -= itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT] - bottomRightEq);


			for (int i = 0; i < 9; i++) {
				if (backLattice[idx].adj[i] < 0.0f) {
					backLattice[idx].adj[i] = 0.0f;
				} else if (backLattice[idx].adj[i] > 1.0f) {
					backLattice[idx].adj[i] = 1.0f;
				}
			}

		}
	}


}



void LBM2D_1D_indices::moveParticles() {


	glm::vec2 adjVelocities[4];

//#pragma omp parallel for/* simd*/
	for (int i = 0; i < particleSystem->numParticles; i++) {
		float x = particleVertices[i].x;
		float y = particleVertices[i].y;

		//printf("OpenMP move particles num threads = %d\n", omp_get_num_threads());

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;

		adjVelocities[0] = velocities[getIdx(leftX, topY)];
		adjVelocities[1] = velocities[getIdx(rightX, topY)];
		adjVelocities[2] = velocities[getIdx(leftX, bottomY)];
		adjVelocities[3] = velocities[getIdx(rightX, bottomY)];

		float horizontalRatio = x - leftX;
		float verticalRatio = y - bottomY;

		glm::vec2 topVelocity = adjVelocities[0] * horizontalRatio + adjVelocities[1] * (1.0f - horizontalRatio);
		glm::vec2 bottomVelocity = adjVelocities[2] * horizontalRatio + adjVelocities[3] * (1.0f - horizontalRatio);

		glm::vec2 finalVelocity = bottomVelocity * verticalRatio + topVelocity * (1.0f - verticalRatio);


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		particleArrows.push_back(particleVertices[i]);
#endif

		if (particleSystem->drawStreamlines) {
			particleSystem->streamLines[i * MAX_STREAMLINE_LENGTH + streamLineCounter] = particleVertices[i];
		}

		particleVertices[i] += glm::vec3(finalVelocity, 0.0f);
#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
		glm::vec3 tmp = particleVertices[i] + 10.0f * glm::vec3(finalVelocity, 0.0f);
		particleArrows.push_back(tmp);
#endif

		if (!respawnLinearly) {
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
				particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1) {
				if (mirrorSides) {
					if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1) {
						particleVertices[i].x = 0.0f;
						particleVertices[i].y = rand2D(i, (int)y) * (latticeHeight - 1);
					} else {
						particleVertices[i].y = (float)((int)(particleVertices[i].y + latticeHeight - 1) % (latticeHeight - 1));
					}
				} else {
					particleVertices[i].x = 0.0f;
					particleVertices[i].y = rand2D(i, (int)y) * (latticeHeight - 1);
				}
				particleVertices[i].z = 0.0f;
			}
		} else {
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
				particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1) {
				if (mirrorSides) {
					if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1) {
						particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
						if (respawnIndex >= respawnMaxY) {
							respawnIndex = respawnMinY;
						}
					} else {
						particleVertices[i] = glm::vec3(x, (int)(particleVertices[i].y + latticeHeight - 1) % (latticeHeight - 1), 0.0f);
					}
				} else {
					particleVertices[i] = glm::vec3(0, respawnIndex++, 0.0f);
					if (respawnIndex >= respawnMaxY) {
						respawnIndex = respawnMinY;
					}
				}

				if (particleSystem->drawStreamlines) {
					for (int k = 0; k < MAX_STREAMLINE_LENGTH; k++) {
						particleSystem->streamLines[i * MAX_STREAMLINE_LENGTH + k] = particleVertices[i];
					}
				}
			}
		}
	}
	streamLineCounter++;
	if (streamLineCounter > MAX_STREAMLINE_LENGTH) {
		streamLineCounter = 0;
	}
}

void LBM2D_1D_indices::updateInlets() {


	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;


	float macroDensity = 1.0f;

	glm::vec3 macroVelocity = inletVelocity;

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


	for (int y = 0; y < latticeHeight; y++) {
		int idx = getIdx(0, y);
		backLattice[idx].adj[DIR_MIDDLE] = middleEq;
		backLattice[idx].adj[DIR_RIGHT] = rightEq;
		backLattice[idx].adj[DIR_TOP] = topEq;
		backLattice[idx].adj[DIR_LEFT] = leftEq;
		backLattice[idx].adj[DIR_BOTTOM] = bottomEq;
		backLattice[idx].adj[DIR_TOP_RIGHT] = topRightEq;
		backLattice[idx].adj[DIR_TOP_LEFT] = topLeftEq;
		backLattice[idx].adj[DIR_BOTTOM_LEFT] = bottomLeftEq;
		backLattice[idx].adj[DIR_BOTTOM_RIGHT] = bottomRightEq;
		for (int i = 0; i < 9; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
		//velocities[idx] = macroVelocity;
	}




}


void LBM2D_1D_indices::updateColliders() {

	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			int idx = getIdx(x, y);

			if (tCol->area[idx]) {

				float right = backLattice[idx].adj[DIR_RIGHT];
				float top = backLattice[idx].adj[DIR_TOP];
				float left = backLattice[idx].adj[DIR_LEFT];
				float bottom = backLattice[idx].adj[DIR_BOTTOM];
				float topRight = backLattice[idx].adj[DIR_TOP_RIGHT];
				float topLeft = backLattice[idx].adj[DIR_TOP_LEFT];
				float bottomLeft = backLattice[idx].adj[DIR_BOTTOM_LEFT];
				float bottomRight = backLattice[idx].adj[DIR_BOTTOM_RIGHT];
				backLattice[idx].adj[DIR_RIGHT] = left;
				backLattice[idx].adj[DIR_TOP] = bottom;
				backLattice[idx].adj[DIR_LEFT] = right;
				backLattice[idx].adj[DIR_BOTTOM] = top;
				backLattice[idx].adj[DIR_TOP_RIGHT] = bottomLeft;
				backLattice[idx].adj[DIR_TOP_LEFT] = bottomRight;
				backLattice[idx].adj[DIR_BOTTOM_LEFT] = topRight;
				backLattice[idx].adj[DIR_BOTTOM_RIGHT] = topLeft;
			}
		}
	}
}

void LBM2D_1D_indices::initBuffers() {


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	vector<glm::vec3> bData;
	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			bData.push_back(glm::vec3(x, y, 0.0f));
		}
	}

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bData.size(), &bData[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

	glBindVertexArray(0);


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

	glBindVertexArray(0);

#endif



}

void LBM2D_1D_indices::initLattice() {
	float weightMiddle = 4.0f / 9.0f;
	float weightAxis = 1.0f / 9.0f;
	float weightDiagonal = 1.0f / 36.0f;

	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			int idx = getIdx(x, y);
			frontLattice[idx].adj[DIR_MIDDLE] = weightMiddle;
			for (int dir = 1; dir <= 4; dir++) {
				frontLattice[idx].adj[dir] = weightAxis;
			}
			for (int dir = 5; dir <= 8; dir++) {
				frontLattice[idx].adj[dir] = weightDiagonal;
			}
		}
	}


}

void LBM2D_1D_indices::precomputeRespawnRange() {

	respawnMinY = 0;
	respawnMaxY = latticeHeight;
	bool minSet = false;
	bool maxSet = false;

	for (int y = 0; y < latticeHeight; y++) {
		if (!minSet && !tCol->area[latticeWidth * y]) {
			respawnMinY = y;
			minSet = true;
		}
		if (minSet && tCol->area[latticeWidth * y]) {
			respawnMaxY = y - 1;
			maxSet = true;
			break;
		}
	}
	if (!minSet && !maxSet) {
		cerr << "The left wall of the scene is completely blocked off! Inlet incorrect" << endl;
		exit(-1);
	}
	if (!maxSet) {
		respawnMaxY = latticeHeight - 1;
	}
	cout << " || min respawn y = " << respawnMinY << ", max respawn y = " << respawnMaxY << endl;

	respawnIndex = respawnMinY;

	cudaMemcpyToSymbol(d_respawnIndex, &respawnIndex, sizeof(int));
	cudaMemcpyToSymbol(d_respawnMinY, &respawnMinY, sizeof(int));
	cudaMemcpyToSymbol(d_respawnMaxY, &respawnMaxY, sizeof(int));


}

void LBM2D_1D_indices::swapLattices() {
	// CPU
	Node *tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;

	// GPU
	tmp = d_frontLattice;
	d_frontLattice = d_backLattice;
	d_backLattice = tmp;

}

float LBM2D_1D_indices::calculateMacroscopicDensity(int x, int y) {

	float macroDensity = 0.0f;
	int idx = getIdx(x, y);
	for (int i = 0; i < 9; i++) {
		macroDensity += backLattice[idx].adj[i];
	}
	return macroDensity;
}

glm::vec3 LBM2D_1D_indices::calculateMacroscopicVelocity(int x, int y, float macroDensity) {
	glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

	int idx = getIdx(x, y);
	macroVelocity += vRight * backLattice[idx].adj[DIR_RIGHT];
	macroVelocity += vTop * backLattice[idx].adj[DIR_TOP];
	macroVelocity += vLeft * backLattice[idx].adj[DIR_LEFT];
	macroVelocity += vBottom * backLattice[idx].adj[DIR_BOTTOM];
	macroVelocity += vTopRight * backLattice[idx].adj[DIR_TOP_RIGHT];
	macroVelocity += vTopLeft * backLattice[idx].adj[DIR_TOP_LEFT];
	macroVelocity += vBottomLeft * backLattice[idx].adj[DIR_BOTTOM_LEFT];
	macroVelocity += vBottomRight * backLattice[idx].adj[DIR_BOTTOM_RIGHT];
	macroVelocity /= macroDensity;


	return macroVelocity;
}
