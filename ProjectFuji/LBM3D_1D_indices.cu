#include "LBM3D_1D_indices.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glm\gtx\norm.hpp>


#include <iostream>
#include "CUDAUtils.cuh"


__constant__ int d_latticeWidth;		///< Lattice width constant on the device
__constant__ int d_latticeHeight;		///< Lattice height constant on the device
__constant__ int d_latticeDepth;		///< Lattice depth constant on the device
__constant__ int d_latticeSize;			///< Lattice size constant on the device (latticeWidth * latticeHeight * latticeDepth)
__constant__ float d_tau;				///< Tau value on the device
__constant__ float d_itau;				///< Inverse tau value (1.0f / tau) on the device
__constant__ int d_mirrorSides;			///< Whether to mirror sides (cycle) on the device



__device__ int d_respawnY = 0;			///< Respawn y coordinate on the device, not used (random respawn now used)
__device__ int d_respawnZ = 0;			///< Respawn z coordinate on the device, not used (random respawn now used)


/// Returns the flattened index using the device constants and provided coordinates.
__device__ int getIdxKer(int x, int y, int z) {
	return (x + d_latticeWidth * (y + d_latticeHeight * z));
}

/// Returns uniform random between 0.0 and 1.0. Provided from different student's work.
__device__ __host__ float rand(int x, int y) {
	int n = x + y * 57;
	n = (n << 13) ^ n;

	return ((1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f) + 1.0f) * 0.5f;
}

/// Maps the value to the viridis color map.
__device__ glm::vec3 mapToViridis3D(float val) {
	val = glm::clamp(val, 0.0f, 1.0f);
	int discreteVal = (int)(val * 255.0f);
	return glm::vec3(viridis_cm[discreteVal][0], viridis_cm[discreteVal][1], viridis_cm[discreteVal][2]);
}


/// Kernel for moving particles that uses OpenGL interoperability.
/**
	Kernel for moving particles that uses OpenGL interoperability for setting particle positions and colors.
	If the particles venture beyond the simulation bounding volume, they are randomly respawned.
	If we use side mirroring (cycling), particles that go beyond side walls (on the z axis) will be mirrored/cycled to the other
	side of the bounding volume.
	\param[in] particleVertices		Vertices (positions stored in VBO) of particles to be updated/moved.
	\param[in] velocities			Array of velocities that will act on the particles.
	\param[in] numParticles			Number of particles.
	\param[in] particleColors		VBO of particle colors.
*/
__global__ void moveParticlesKernelInterop(glm::vec3 *particleVertices, glm::vec3 *velocities, int *numParticles, glm::vec3 *particleColors) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	glm::vec3 adjVelocities[8];

	while (idx < *numParticles) {

		float x = particleVertices[idx].x;
		float y = particleVertices[idx].y;
		float z = particleVertices[idx].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[getIdxKer(leftX, topY, backZ)];
		adjVelocities[1] = velocities[getIdxKer(rightX, topY, backZ)];
		adjVelocities[2] = velocities[getIdxKer(leftX, bottomY, backZ)];
		adjVelocities[3] = velocities[getIdxKer(rightX, bottomY, backZ)];
		adjVelocities[4] = velocities[getIdxKer(leftX, topY, frontZ)];
		adjVelocities[5] = velocities[getIdxKer(rightX, topY, frontZ)];
		adjVelocities[6] = velocities[getIdxKer(leftX, bottomY, frontZ)];
		adjVelocities[7] = velocities[getIdxKer(rightX, bottomY, frontZ)];

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

		particleVertices[idx].x += finalVelocity.x;
		particleVertices[idx].y += finalVelocity.y;
		particleVertices[idx].z += finalVelocity.z;


		particleColors[idx] = mapToViridis3D(glm::length2(finalVelocity) * 4.0f);


		
		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
			particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 ||
			particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {
			
			particleVertices[idx].x = 0.0f;
			//particleVertices[idx].y = y;
			particleVertices[idx].y = rand(idx, y) * (d_latticeHeight - 1);
			//particleVertices[idx].z = z;
			particleVertices[idx].z = rand(idx, z) * (d_latticeDepth - 1);
			//particleVertices[idx].y = d_respawnY;
			//particleVertices[idx].z = d_respawnZ++;

		}
		
		

		/*
		if (d_mirrorSides && (particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1)) {
			particleVertices[idx].z = (int)(particleVertices[idx].z + d_latticeDepth - 1) % (d_latticeDepth - 1);
		} else if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
					particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 || 
				   particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {
			particleVertices[idx].x = 0.0f;
			particleVertices[idx].y = d_respawnY;
			particleVertices[idx].z = d_respawnZ;
			//d_respawnZ++;
			atomicAdd(&d_respawnZ, 1);

			if (d_respawnZ >= d_latticeDepth - 1) {
				d_respawnZ = 0;
				//atomicExch(&d_respawnZ, 0);
				//d_respawnY++;
				atomicAdd(&d_respawnY, 1);
			}
			if (d_respawnY >= d_latticeHeight - 1) {
				d_respawnY = 0;
				//atomicExch(&d_respawnY, 0);
			}
		}
		*/
		

		/*
		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
			particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 ||
			particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {

			if (d_mirrorSides) {
				if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1 ||
					particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1) {
					particleVertices[idx].x = 0.0f;
					particleVertices[idx].y = d_respawnY;
					particleVertices[idx].z = d_respawnZ++;

					if (d_respawnZ >= d_latticeDepth - 1) {
						d_respawnZ = 0;
						d_respawnY++;
					}
					if (d_respawnY >= d_latticeHeight - 1) {
						d_respawnY = 0;
					}
				} else {
					//particleVertices[idx].x = x;
					//particleVertices[idx].y = y;
					particleVertices[idx].z = (int)(particleVertices[idx].z + d_latticeDepth - 1) % (d_latticeDepth - 1);
				}
			} else {


				//particleVertices[idx] = glm::vec3(0.0f, d_respawnY, d_respawnZ++);
				particleVertices[idx].x = 0.0f;
				particleVertices[idx].y = d_respawnY;
				particleVertices[idx].z = d_respawnZ++;

				if (d_respawnZ >= d_latticeDepth - 1) {
					d_respawnZ = 0;
					d_respawnY++;
				}
				if (d_respawnY >= d_latticeHeight - 1) {
					d_respawnY = 0;
				}
			}
		}
		*/
		
		

/*
		if (particleVertices[idx].x <= 0.0f || particleVertices[idx].x >= d_latticeWidth - 1) {
			particleVertices[idx].x = 0.0f;
		} else if (particleVertices[idx].y <= 0.0f || particleVertices[idx].y >= d_latticeHeight - 1 ||
				   particleVertices[idx].z <= 0.0f || particleVertices[idx].z >= d_latticeDepth - 1) {

			particleVertices[idx].y = (float)((int)(particleVertices[idx].y + d_latticeHeight - 1) % (d_latticeHeight - 1));
			particleVertices[idx].z = (float)((int)(particleVertices[idx].z + d_latticeDepth - 1) % (d_latticeDepth - 1));
		}*/

		idx += blockDim.x * blockDim.y * gridDim.x;

	}
}

/// Kernel for clearing the back lattice.
/**
	Kernel that clears the back lattice.
	\param[in] backLattice	Pointer to the back lattice to be cleared.
*/
__global__ void clearBackLatticeKernel(Node3D *backLattice) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	//if (idx == 0) {
	//	printf("d_latticeSize = %d\n", d_latticeSize);
	//}

	if (idx < d_latticeSize) {
		for (int i = 0; i < 19; i++) {
			backLattice[idx].adj[i] = 0.0f;
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
__global__ void updateInletsKernel(Node3D *backLattice, glm::vec3 *velocities, glm::vec3 inletVelocity) {

	float macroDensity = 1.0f;
	//glm::vec3 macroVelocity = inletVelocity;


	float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
	float leftTermAxis = WEIGHT_AXIS * macroDensity;
	float leftTermNonaxial = WEIGHT_NON_AXIAL * macroDensity;


	float macroVelocityDot = glm::dot(inletVelocity, inletVelocity);
	float thirdTerm = 1.5f * macroVelocityDot;

	float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

	float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], inletVelocity);
	float firstTerm = 3.0f * dotProd;
	float secondTerm = 4.5f * dotProd * dotProd;
	float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], inletVelocity);
	firstTerm = 3.0f * dotProd;
	secondTerm = 4.5f * dotProd * dotProd;
	float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;


	int x = idx % d_latticeWidth;
	//int y = (idx / d_latticeWidth) % d_latticeHeight;
	//int z = idx / (d_latticeHeight * d_latticeWidth);


	if (x == 0 && idx < d_latticeSize) {
		backLattice[idx].adj[DIR_MIDDLE_VERTEX] = middleEq;
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


/// Kernel for calculating the collision operator.
/**
	Kernel that calculates the collision operator using Bhatnagar-Gross-Krook operator.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
*/
__global__ void collisionStepKernel(Node3D *backLattice, glm::vec3 *velocities) {
	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {
		float macroDensity = 0.0f;
		for (int i = 0; i < 19; i++) {
			macroDensity += backLattice[idx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
		macroVelocity += dirVectorsConst[DIR_LEFT_FACE] * backLattice[idx].adj[DIR_LEFT_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_FACE] * backLattice[idx].adj[DIR_FRONT_FACE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FACE] * backLattice[idx].adj[DIR_BOTTOM_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_LEFT_EDGE] * backLattice[idx].adj[DIR_FRONT_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BACK_LEFT_EDGE] * backLattice[idx].adj[DIR_BACK_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_LEFT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_LEFT_EDGE] * backLattice[idx].adj[DIR_TOP_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FRONT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_FRONT_EDGE] * backLattice[idx].adj[DIR_TOP_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_RIGHT_FACE] * backLattice[idx].adj[DIR_RIGHT_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_FACE] * backLattice[idx].adj[DIR_BACK_FACE];
		macroVelocity += dirVectorsConst[DIR_TOP_FACE] * backLattice[idx].adj[DIR_TOP_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_RIGHT_EDGE] * backLattice[idx].adj[DIR_BACK_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_FRONT_RIGHT_EDGE] * backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_RIGHT_EDGE] * backLattice[idx].adj[DIR_TOP_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE] * backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_BACK_EDGE] * backLattice[idx].adj[DIR_TOP_BACK_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_BACK_EDGE] * backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE];
		macroVelocity /= macroDensity;

		velocities[idx] = macroVelocity;

		float leftTermMiddle = weightMiddle * macroDensity;
		float leftTermAxis = weightAxis * macroDensity;
		float leftTermNonaxial = weightNonaxial * macroDensity;

		float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
		float thirdTerm = 1.5f * macroVelocityDot;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], macroVelocity);
		float firstTerm = 3.0f * dotProd;
		float secondTerm = 4.5f * dotProd * dotProd;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		backLattice[idx].adj[DIR_MIDDLE_VERTEX] -= d_itau * (backLattice[idx].adj[DIR_MIDDLE_VERTEX] - middleEq);
		backLattice[idx].adj[DIR_RIGHT_FACE] -= d_itau * (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
		backLattice[idx].adj[DIR_LEFT_FACE] -= d_itau * (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
		backLattice[idx].adj[DIR_BACK_FACE] -= d_itau * (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
		backLattice[idx].adj[DIR_FRONT_FACE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
		backLattice[idx].adj[DIR_TOP_FACE] -= d_itau * (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
		backLattice[idx].adj[DIR_BOTTOM_FACE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
		backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
		backLattice[idx].adj[DIR_BACK_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
		backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
		backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
		backLattice[idx].adj[DIR_TOP_BACK_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
		backLattice[idx].adj[DIR_TOP_FRONT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
		backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
		backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
		backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
		backLattice[idx].adj[DIR_TOP_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
		backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
		backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] -= d_itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


		for (int i = 0; i < 19; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}


}

/// Kernel for calculating the collision operator that uses the shared memory (in naive manner).
/**
	Kernel that calculates the collision operator using Bhatnagar-Gross-Krook operator.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
*/
__global__ void collisionStepKernelShared(Node3D *backLattice, glm::vec3 *velocities) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	extern __shared__ Node3D cache[];
	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;


	if (idx < d_latticeSize) {

		cache[cacheIdx] = backLattice[idx];
		//__syncthreads(); // not needed

		float macroDensity = 0.0f;
		for (int i = 0; i < 19; i++) {
			macroDensity += cache[cacheIdx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		//macroVelocity += vMiddle * backLattice[idx].adj[DIR_MIDDLE];
		macroVelocity += dirVectorsConst[DIR_LEFT_FACE] * cache[cacheIdx].adj[DIR_LEFT_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_FACE] * cache[cacheIdx].adj[DIR_FRONT_FACE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FACE] * cache[cacheIdx].adj[DIR_BOTTOM_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_LEFT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BACK_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_LEFT_EDGE] * cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FRONT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_FRONT_EDGE] * cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_RIGHT_FACE] * cache[cacheIdx].adj[DIR_RIGHT_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_FACE] * cache[cacheIdx].adj[DIR_BACK_FACE];
		macroVelocity += dirVectorsConst[DIR_TOP_FACE] * cache[cacheIdx].adj[DIR_TOP_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_FRONT_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_BACK_EDGE] * cache[cacheIdx].adj[DIR_TOP_BACK_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_BACK_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE];
		macroVelocity /= macroDensity;

		velocities[idx] = macroVelocity;

		float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
		float leftTermAxis = WEIGHT_AXIS * macroDensity;
		float leftTermNonaxial = WEIGHT_NON_AXIAL * macroDensity;

		float macroVelocityDot = glm::dot(macroVelocity, macroVelocity);
		float thirdTerm = 1.5f * macroVelocityDot;

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		float dotProd = glm::dot(dirVectorsConst[DIR_RIGHT_FACE], macroVelocity);
		float firstTerm = 3.0f * dotProd;
		float secondTerm = 4.5f * dotProd * dotProd;
		float rightEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_LEFT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float leftEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FACE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomEq = leftTermAxis + leftTermAxis * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BACK_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float backLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_FRONT_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float frontLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_BACK_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomBackEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_FRONT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomFrontEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_TOP_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float topLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomRightEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);

		dotProd = glm::dot(dirVectorsConst[DIR_BOTTOM_LEFT_EDGE], macroVelocity);
		firstTerm = 3.0f * dotProd;
		secondTerm = 4.5f * dotProd * dotProd;
		float bottomLeftEq = leftTermNonaxial + leftTermNonaxial * (firstTerm + secondTerm - thirdTerm);


		cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] -= d_itau * (cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] - middleEq);
		cache[cacheIdx].adj[DIR_RIGHT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_RIGHT_FACE] - rightEq);
		cache[cacheIdx].adj[DIR_LEFT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_LEFT_FACE] - leftEq);
		cache[cacheIdx].adj[DIR_BACK_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_FACE] - backEq);
		cache[cacheIdx].adj[DIR_FRONT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_FACE] - frontEq);
		cache[cacheIdx].adj[DIR_TOP_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FACE] - topEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FACE] - bottomEq);
		cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
		cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
		cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
		cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
		cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
		cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
		cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
		cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
		cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
		cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


		for (int i = 0; i < 19; i++) {
			if (cache[cacheIdx].adj[i] < 0.0f) {
				cache[cacheIdx].adj[i] = 0.0f;
			} else if (cache[cacheIdx].adj[i] > 1.0f) {
				cache[cacheIdx].adj[i] = 1.0f;
			}
		}

		backLattice[idx] = cache[cacheIdx];

	}
}


/// Kernel for calculating the collision operator using shared memory with smaller register usage.
/**
	Kernel that calculates the collision operator using Bhatnagar-Gross-Krook operator.
	Uses shared memory and less registers. Slower than its naive version unfortunately.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
*/
__global__ void collisionStepKernelStreamlinedShared(Node3D *backLattice, glm::vec3 *velocities) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	extern __shared__ Node3D cache[];
	int cacheIdx = threadIdx.x + blockDim.x * threadIdx.y;


	if (idx < d_latticeSize) {

		cache[cacheIdx] = backLattice[idx];

		float macroDensity = 0.0f;
		for (int i = 0; i < 19; i++) {
			macroDensity += cache[cacheIdx].adj[i];
		}

		glm::vec3 macroVelocity = glm::vec3(0.0f, 0.0f, 0.0f);

		macroVelocity += dirVectorsConst[DIR_LEFT_FACE] * cache[cacheIdx].adj[DIR_LEFT_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_FACE] * cache[cacheIdx].adj[DIR_FRONT_FACE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FACE] * cache[cacheIdx].adj[DIR_BOTTOM_FACE];
		macroVelocity += dirVectorsConst[DIR_FRONT_LEFT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BACK_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_LEFT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_LEFT_EDGE] * cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_FRONT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_FRONT_EDGE] * cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE];
		macroVelocity += dirVectorsConst[DIR_RIGHT_FACE] * cache[cacheIdx].adj[DIR_RIGHT_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_FACE] * cache[cacheIdx].adj[DIR_BACK_FACE];
		macroVelocity += dirVectorsConst[DIR_TOP_FACE] * cache[cacheIdx].adj[DIR_TOP_FACE];
		macroVelocity += dirVectorsConst[DIR_BACK_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_FRONT_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_RIGHT_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE];
		macroVelocity += dirVectorsConst[DIR_TOP_BACK_EDGE] * cache[cacheIdx].adj[DIR_TOP_BACK_EDGE];
		macroVelocity += dirVectorsConst[DIR_BOTTOM_BACK_EDGE] * cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE];
		macroVelocity /= macroDensity;

		velocities[idx] = macroVelocity;

		float leftTermMiddle = WEIGHT_MIDDLE * macroDensity;
		float leftTermAxis = WEIGHT_AXIS * macroDensity;
		float leftTermNonaxial = WEIGHT_NON_AXIAL * macroDensity;

		float thirdTerm = 1.5f * glm::dot(macroVelocity, macroVelocity);

		float middleEq = leftTermMiddle + leftTermMiddle * (-thirdTerm);

		float tmp;

		float rightEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);

		float leftEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.x + 4.5f * macroVelocity.x * macroVelocity.x - thirdTerm);

		float frontEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.z + 4.5f * macroVelocity.z * macroVelocity.z - thirdTerm);

		float backEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.z + 4.5f * macroVelocity.z * macroVelocity.z - thirdTerm);

		float topEq = leftTermAxis * (1.0f + 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);

		float bottomEq = leftTermAxis * (1.0f - 3.0f * macroVelocity.y + 4.5f * macroVelocity.y * macroVelocity.y - thirdTerm);

		tmp = macroVelocity.x - macroVelocity.z;
		float backRightEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = macroVelocity.x + macroVelocity.z;
		float backLeftEq = leftTermNonaxial * (1.0f - 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		float frontRightEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = -macroVelocity.x + macroVelocity.z;
		float frontLeftEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = macroVelocity.y - macroVelocity.z;
		float topBackEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = macroVelocity.y + macroVelocity.z;
		float topFrontEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = -macroVelocity.y - macroVelocity.z;
		float bottomBackEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = -macroVelocity.y + macroVelocity.z;
		float bottomFrontEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = macroVelocity.x + macroVelocity.y;
		float topRightEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = -macroVelocity.x + macroVelocity.y;
		float topLeftEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = macroVelocity.x - macroVelocity.y;
		float bottomRightEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);

		tmp = -macroVelocity.x - macroVelocity.y;
		float bottomLeftEq = leftTermNonaxial * (1.0f + 3.0f * tmp + 4.5f * tmp * tmp - thirdTerm);


		cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] -= d_itau * (cache[cacheIdx].adj[DIR_MIDDLE_VERTEX] - middleEq);
		cache[cacheIdx].adj[DIR_RIGHT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_RIGHT_FACE] - rightEq);
		cache[cacheIdx].adj[DIR_LEFT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_LEFT_FACE] - leftEq);
		cache[cacheIdx].adj[DIR_BACK_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_FACE] - backEq);
		cache[cacheIdx].adj[DIR_FRONT_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_FACE] - frontEq);
		cache[cacheIdx].adj[DIR_TOP_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FACE] - topEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FACE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FACE] - bottomEq);
		cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
		cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
		cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
		cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
		cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
		cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
		cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
		cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
		cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
		cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
		cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
		cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] -= d_itau * (cache[cacheIdx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


		for (int i = 0; i < 19; i++) {
			if (cache[cacheIdx].adj[i] < 0.0f) {
				cache[cacheIdx].adj[i] = 0.0f;
			} else if (cache[cacheIdx].adj[i] > 1.0f) {
				cache[cacheIdx].adj[i] = 1.0f;
			}
		}

		backLattice[idx] = cache[cacheIdx];

	}
}


/// Kernel for updating colliders/obstacles in the lattice.
/**
	Updates colliders/obstacles by using the full bounce back approach.
	\param[in] backLattice		Back lattice in which we do our calculations.
	\param[in] velocities		Velocities array for the lattice.
	\param[in] heightMap		Height map of the scene.
*/
__global__ void updateCollidersKernel(Node3D *backLattice, glm::vec3 *velocities, float *heightMap) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {

		int x = idx % d_latticeWidth;
		int y = (idx / d_latticeWidth) % d_latticeHeight;
		int z = idx / (d_latticeHeight * d_latticeWidth);

		//float tmp;

		if ((heightMap[x + z * d_latticeWidth] >= y && heightMap[x + z * d_latticeWidth] > 0.01f) || y == 0) {

			// possible way of using less local memory
			//tmp = backLattice[idx].adj[DIR_RIGHT_FACE];
			//backLattice[idx].adj[DIR_RIGHT_FACE] = backLattice[idx].adj[DIR_LEFT_FACE];
			//backLattice[idx].adj[DIR_LEFT_FACE] = tmp;
			//tmp = backLattice[idx].adj[DIR_BACK_FACE];


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

		}
	}
}


/// Kernel that streams the microscopic particles from the previous frame.
/**
	 Kernel that streams the microscopic particles from the previous frame.
	 \param[in] backLatice		Lattice that will be used in the current frame (the one we are currently updating).
	 \param[in] frontLattice	Lattice from the previous frame from which we stream the particles.
*/
__global__ void streamingStepKernel(Node3D *backLattice, Node3D *frontLattice) {

	int idx = threadIdx.x + blockDim.x * threadIdx.y; // idx in block
	idx += blockDim.x * blockDim.y * blockIdx.x;

	if (idx < d_latticeSize) {

		int x = idx % d_latticeWidth;
		int y = (idx / d_latticeWidth) % d_latticeHeight;
		int z = idx / (d_latticeHeight * d_latticeWidth);

		backLattice[idx].adj[DIR_MIDDLE_VERTEX] += frontLattice[idx].adj[DIR_MIDDLE_VERTEX];

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
		if (front > d_latticeDepth - 1) {
			front = d_latticeDepth - 1;
		}
		if (back < 0) {
			back = 0;
		}

		backLattice[idx].adj[DIR_LEFT_FACE] += frontLattice[getIdxKer(right, y, z)].adj[DIR_LEFT_FACE];
		backLattice[idx].adj[DIR_FRONT_FACE] += frontLattice[getIdxKer(x, y, back)].adj[DIR_FRONT_FACE];
		backLattice[idx].adj[DIR_BOTTOM_FACE] += frontLattice[getIdxKer(x, top, z)].adj[DIR_BOTTOM_FACE];
		backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] += frontLattice[getIdxKer(right, y, back)].adj[DIR_FRONT_LEFT_EDGE];
		backLattice[idx].adj[DIR_BACK_LEFT_EDGE] += frontLattice[getIdxKer(right, y, front)].adj[DIR_BACK_LEFT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] += frontLattice[getIdxKer(right, top, z)].adj[DIR_BOTTOM_LEFT_EDGE];
		backLattice[idx].adj[DIR_TOP_LEFT_EDGE] += frontLattice[getIdxKer(right, bottom, z)].adj[DIR_TOP_LEFT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] += frontLattice[getIdxKer(x, top, back)].adj[DIR_BOTTOM_FRONT_EDGE];
		backLattice[idx].adj[DIR_TOP_FRONT_EDGE] += frontLattice[getIdxKer(x, bottom, back)].adj[DIR_TOP_FRONT_EDGE];
		backLattice[idx].adj[DIR_RIGHT_FACE] += frontLattice[getIdxKer(left, y, z)].adj[DIR_RIGHT_FACE];
		backLattice[idx].adj[DIR_BACK_FACE] += frontLattice[getIdxKer(x, y, front)].adj[DIR_BACK_FACE];
		backLattice[idx].adj[DIR_TOP_FACE] += frontLattice[getIdxKer(x, bottom, z)].adj[DIR_TOP_FACE];
		backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] += frontLattice[getIdxKer(left, y, front)].adj[DIR_BACK_RIGHT_EDGE];
		backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] += frontLattice[getIdxKer(left, y, back)].adj[DIR_FRONT_RIGHT_EDGE];
		backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] += frontLattice[getIdxKer(left, bottom, z)].adj[DIR_TOP_RIGHT_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] += frontLattice[getIdxKer(left, top, z)].adj[DIR_BOTTOM_RIGHT_EDGE];
		backLattice[idx].adj[DIR_TOP_BACK_EDGE] += frontLattice[getIdxKer(x, bottom, front)].adj[DIR_TOP_BACK_EDGE];
		backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] += frontLattice[getIdxKer(x, top, front)].adj[DIR_BOTTOM_BACK_EDGE];

		for (int i = 0; i < 19; i++) {
			if (backLattice[idx].adj[i] < 0.0f) {
				backLattice[idx].adj[i] = 0.0f;
			} else if (backLattice[idx].adj[i] > 1.0f) {
				backLattice[idx].adj[i] = 1.0f;
			}
		}
	}

}




LBM3D_1D_indices::LBM3D_1D_indices() {
}




LBM3D_1D_indices::LBM3D_1D_indices(glm::ivec3 dim, string sceneFilename, float tau, ParticleSystem *particleSystem, dim3 blockDim)
	: LBM(dim, sceneFilename, tau, particleSystem), blockDim(blockDim) {

	initScene();


	frontLattice = new Node3D[latticeSize]();
	backLattice = new Node3D[latticeSize]();
	velocities = new glm::vec3[latticeSize]();

	cudaMalloc((void**)&d_frontLattice, sizeof(Node3D) * latticeSize);
	cudaMalloc((void**)&d_backLattice, sizeof(Node3D) * latticeSize);
	cudaMalloc((void**)&d_velocities, sizeof(glm::vec3) * latticeSize);


	cudaGraphicsGLRegisterBuffer(&cudaParticleVerticesVBO, particleSystem->vbo, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&cudaParticleColorsVBO, particleSystem->colorsVBO, cudaGraphicsMapFlagsWriteDiscard);


	cudaMemcpyToSymbol(dirVectorsConst, &directionVectors3D[0], 19 * sizeof(glm::vec3));

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;
	cudaMemcpyToSymbol(WEIGHT_MIDDLE, &weightMiddle, sizeof(float));
	cudaMemcpyToSymbol(WEIGHT_AXIS, &weightAxis, sizeof(float));
	cudaMemcpyToSymbol(WEIGHT_NON_AXIAL, &weightNonaxial, sizeof(float));


	cudaMemcpyToSymbol(d_latticeWidth, &latticeWidth, sizeof(int));
	cudaMemcpyToSymbol(d_latticeHeight, &latticeHeight, sizeof(int));
	cudaMemcpyToSymbol(d_latticeDepth, &latticeDepth, sizeof(int));
	cudaMemcpyToSymbol(d_latticeSize, &latticeSize, sizeof(int));
	cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_itau, &itau, sizeof(float));
	cudaMemcpyToSymbol(d_mirrorSides, &mirrorSides, sizeof(int));


	gridDim = dim3((unsigned int)ceil(latticeSize / (blockDim.x * blockDim.y * blockDim.z)) + 1, 1, 1);
	cacheSize = blockDim.x * blockDim.y * blockDim.z * sizeof(Node3D);


	initBuffers();
	initLattice();

	cudaMemcpy(d_backLattice, backLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec3) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);


}


LBM3D_1D_indices::~LBM3D_1D_indices() {

	delete[] frontLattice;
	delete[] backLattice;
	delete[] velocities;

	delete heightMap;

	cudaFree(d_frontLattice);
	cudaFree(d_backLattice);
	cudaFree(d_velocities);

	cudaGraphicsUnregisterResource(cudaParticleVerticesVBO);
	cudaGraphicsUnregisterResource(cudaParticleColorsVBO);


}

void LBM3D_1D_indices::recalculateVariables() {
	LBM::recalculateVariables();
	cudaMemcpyToSymbol(d_tau, &tau, sizeof(float));
	cudaMemcpyToSymbol(d_itau, &itau, sizeof(float));
}

void LBM3D_1D_indices::initScene() {
	heightMap = new HeightMap(sceneFilename, latticeHeight, nullptr);


	latticeWidth = heightMap->width;
	latticeDepth = heightMap->height;
	latticeSize = latticeWidth * latticeHeight * latticeDepth;

	float *tempHM = new float[latticeWidth * latticeDepth];
	for (int z = 0; z < latticeDepth; z++) {
		for (int x = 0; x < latticeWidth; x++) {
			tempHM[x + z * latticeWidth] = heightMap->data[x][z];
		}
	}
	cudaMalloc((void**)&d_heightMap, sizeof(float) * latticeWidth * latticeDepth);
	//cudaMemcpy(d_heightMap, heightMap->data, sizeof(float) * latticeWidth * latticeDepth, cudaMemcpyHostToDevice);
	cudaMemcpy(d_heightMap, tempHM, sizeof(float) * latticeWidth * latticeDepth, cudaMemcpyHostToDevice);


	cout << "lattice width = " << latticeWidth << ", height = " << latticeHeight << ", depth = " << latticeDepth << endl;

	delete[] tempHM;

	particleVertices = particleSystem->particleVertices;
	d_numParticles = particleSystem->d_numParticles;

	particleSystem->initParticlePositions(latticeWidth, latticeHeight, latticeDepth, heightMap);


}

void LBM3D_1D_indices::draw(ShaderProgram & shader) {

#ifdef DRAW_VELOCITY_ARROWS
	shader.setVec3("uColor", glm::vec3(0.2f, 0.3f, 1.0f));
	glBindVertexArray(velocityVAO);
	glBindBuffer(GL_ARRAY_BUFFER, velocityVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * velocityArrows.size(), &velocityArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, velocityArrows.size());
#endif


#ifdef DRAW_PARTICLE_VELOCITY_ARROWS
	shader.setVec3("uColor", glm::vec3(0.8f, 1.0f, 0.6f));

	glBindVertexArray(particleArrowsVAO);

	glBindBuffer(GL_ARRAY_BUFFER, particleArrowsVBO);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * particleArrows.size(), &particleArrows[0], GL_STATIC_DRAW);
	glDrawArrays(GL_LINES, 0, particleArrows.size());
#endif

	heightMap->draw();

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

void LBM3D_1D_indices::doStepCUDA() {

	//CHECK_ERROR(cudaPeekAtLastError());

	// ============================================= clear back lattice CUDA
	clearBackLatticeKernel << <gridDim, blockDim >> > (d_backLattice);
	//CHECK_ERROR(cudaPeekAtLastError());

	// ============================================= update inlets CUDA
	updateInletsKernel << <gridDim, blockDim >> > (d_backLattice, d_velocities, inletVelocity);
	//CHECK_ERROR(cudaPeekAtLastError());

	// ============================================= streaming step CUDA
	streamingStepKernel << <gridDim, blockDim >> > (d_backLattice, d_frontLattice);
	//CHECK_ERROR(cudaPeekAtLastError());

	// ============================================= update colliders CUDA
	updateCollidersKernel << <gridDim, blockDim >> > (d_backLattice, d_velocities, d_heightMap);
	//CHECK_ERROR(cudaPeekAtLastError());

	// ============================================= collision step CUDA
	//collisionStepKernel << <gridDim, blockDim >> > (d_backLattice, d_velocities);
	collisionStepKernelShared << <gridDim, blockDim, cacheSize >> > (d_backLattice, d_velocities);
	//collisionStepKernelStreamlinedShared << <gridDim, blockDim, cacheSize >> > (d_backLattice, d_velocities);

	//CHECK_ERROR(cudaPeekAtLastError());
	

	// ============================================= move particles CUDA - different respawn from CPU !!!

	glm::vec3 *d_particleVerticesVBO;
	cudaGraphicsMapResources(1, &cudaParticleVerticesVBO, 0);
	//CHECK_ERROR(cudaPeekAtLastError());

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&d_particleVerticesVBO, &num_bytes, cudaParticleVerticesVBO);
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	glm::vec3 *d_particleColorsVBO;
	cudaGraphicsMapResources(1, &cudaParticleColorsVBO, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_particleColorsVBO, &num_bytes, cudaParticleColorsVBO);

	moveParticlesKernelInterop << <gridDim, blockDim >> > (d_particleVerticesVBO, d_velocities, d_numParticles, d_particleColorsVBO);
	//CHECK_ERROR(cudaPeekAtLastError());

	cudaGraphicsUnmapResources(1, &cudaParticleVerticesVBO, 0);
	cudaGraphicsUnmapResources(1, &cudaParticleColorsVBO, 0);

	//CHECK_ERROR(cudaPeekAtLastError());




	swapLattices();
	//CHECK_ERROR(cudaPeekAtLastError());

	frameId++;

}








void LBM3D_1D_indices::clearBackLattice() {
	for (int i = 0; i < latticeSize; i++) {
		for (int j = 0; j < 19; j++) {
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

void LBM3D_1D_indices::streamingStep() {


	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);
				backLattice[idx].adj[DIR_MIDDLE_VERTEX] += frontLattice[idx].adj[DIR_MIDDLE_VERTEX];

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
				if (front > latticeDepth - 1) {
					front = latticeDepth - 1;
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

	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {

				int idx = getIdx(x, y, z);

				float macroDensity = calculateMacroscopicDensity(x, y, z);
				glm::vec3 macroVelocity = calculateMacroscopicVelocity(x, y, z, macroDensity);

				velocities[idx] = macroVelocity;

#ifdef DRAW_VELOCITY_ARROWS
				velocityArrows.push_back(glm::vec3(x, y, z));
				velocityArrows.push_back(glm::vec3(x, y, z) + velocities[idx] * 2.0f);
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


				if (useSubgridModel) {
					// SUBGRID MODEL - EXPERIMENTAL - GIVES INCORRECT VALUES
					float tensor[3][3];

					float diffs[19];
					diffs[0] = (backLattice[idx].adj[DIR_MIDDLE_VERTEX] - middleEq);
					diffs[1] = (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
					diffs[2] = (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
					diffs[3] = (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
					diffs[4] = (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
					diffs[5] = (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
					diffs[6] = (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
					diffs[7] = (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
					diffs[8] = (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
					diffs[9] = (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
					diffs[10] = (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
					diffs[11] = (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
					diffs[12] = (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
					diffs[13] = (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
					diffs[14] = (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
					diffs[15] = (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
					diffs[16] = (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
					diffs[17] = (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
					diffs[18] = (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);

					float sum = 0.0f;
					for (int i = 0; i < 19; i++) {
						sum += diffs[i];
					}

					for (int i = 0; i < 9; i++) {
						tensor[0][0] = 0.0f;
					}
					for (int i = 0; i < 19; i++) {
						tensor[0][0] += directionVectors3D[i].x * directionVectors3D[i].x * diffs[i];
						tensor[0][1] += directionVectors3D[i].x * directionVectors3D[i].y * diffs[i];
						tensor[0][2] += directionVectors3D[i].x * directionVectors3D[i].z * diffs[i];
						tensor[1][0] += directionVectors3D[i].y * directionVectors3D[i].x * diffs[i];
						tensor[1][1] += directionVectors3D[i].y * directionVectors3D[i].y * diffs[i];
						tensor[1][2] += directionVectors3D[i].y * directionVectors3D[i].z * diffs[i];
						tensor[2][0] += directionVectors3D[i].z * directionVectors3D[i].x * diffs[i];
						tensor[2][1] += directionVectors3D[i].z * directionVectors3D[i].y * diffs[i];
						tensor[2][2] += directionVectors3D[i].z * directionVectors3D[i].z * diffs[i];
					}

					sum = 0.0f;
					for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
							sum += tensor[i][j] * tensor[i][j];
						}
					}

					float S = (-nu + sqrtf(nu * nu + 18.0f * SMAG_C * sqrtf(sum))) / (6.0f * SMAG_C * SMAG_C);

					tau = 3.0f * (nu + SMAG_C * SMAG_C * S) + 0.5f;
					itau = 1.0f / tau;
				}




				backLattice[idx].adj[DIR_MIDDLE_VERTEX] -= itau * (backLattice[idx].adj[DIR_MIDDLE_VERTEX] - middleEq);
				backLattice[idx].adj[DIR_RIGHT_FACE] -= itau * (backLattice[idx].adj[DIR_RIGHT_FACE] - rightEq);
				backLattice[idx].adj[DIR_LEFT_FACE] -= itau * (backLattice[idx].adj[DIR_LEFT_FACE] - leftEq);
				backLattice[idx].adj[DIR_BACK_FACE] -= itau * (backLattice[idx].adj[DIR_BACK_FACE] - backEq);
				backLattice[idx].adj[DIR_FRONT_FACE] -= itau * (backLattice[idx].adj[DIR_FRONT_FACE] - frontEq);
				backLattice[idx].adj[DIR_TOP_FACE] -= itau * (backLattice[idx].adj[DIR_TOP_FACE] - topEq);
				backLattice[idx].adj[DIR_BOTTOM_FACE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_FACE] - bottomEq);
				backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_BACK_RIGHT_EDGE] - backRightEq);
				backLattice[idx].adj[DIR_BACK_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_BACK_LEFT_EDGE] - backLeftEq);
				backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_FRONT_RIGHT_EDGE] - frontRightEq);
				backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_FRONT_LEFT_EDGE] - frontLeftEq);
				backLattice[idx].adj[DIR_TOP_BACK_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_BACK_EDGE] - topBackEq);
				backLattice[idx].adj[DIR_TOP_FRONT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_FRONT_EDGE] - topFrontEq);
				backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_BACK_EDGE] - bottomBackEq);
				backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_FRONT_EDGE] - bottomFrontEq);
				backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_RIGHT_EDGE] - topRightEq);
				backLattice[idx].adj[DIR_TOP_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_TOP_LEFT_EDGE] - topLeftEq);
				backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_RIGHT_EDGE] - bottomRightEq);
				backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] -= itau * (backLattice[idx].adj[DIR_BOTTOM_LEFT_EDGE] - bottomLeftEq);


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
	for (int i = 0; i < particleSystem->numParticles; i++) {

		float x = particleVertices[i].x;
		float y = particleVertices[i].y;
		float z = particleVertices[i].z;

		int leftX = (int)x;
		int rightX = leftX + 1;
		int bottomY = (int)y;
		int topY = bottomY + 1;
		int backZ = (int)z;
		int frontZ = backZ + 1;

		adjVelocities[0] = velocities[getIdx(leftX, topY, backZ)];
		adjVelocities[1] = velocities[getIdx(rightX, topY, backZ)];
		adjVelocities[2] = velocities[getIdx(leftX, bottomY, backZ)];
		adjVelocities[3] = velocities[getIdx(rightX, bottomY, backZ)];
		adjVelocities[4] = velocities[getIdx(leftX, topY, frontZ)];
		adjVelocities[5] = velocities[getIdx(rightX, topY, frontZ)];
		adjVelocities[6] = velocities[getIdx(leftX, bottomY, frontZ)];
		adjVelocities[7] = velocities[getIdx(rightX, bottomY, frontZ)];

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

		if (!respawnLinearly) {
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
				particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1 ||
				particleVertices[i].z <= 0.0f || particleVertices[i].z >= latticeDepth - 1) {

				particleVertices[i].x = 0.0f;
				particleVertices[i].y = rand(i, (int)y) * (latticeHeight - 1);
				particleVertices[i].z = rand(i, (int)z) * (latticeDepth - 1);
				//particleVertices[i].y = std::rand() % (latticeHeight - 1);
				//particleVertices[i].z = std::rand() % (latticeDepth - 1);


			}
		} else {
			if (particleVertices[i].x <= 0.0f || particleVertices[i].x >= latticeWidth - 1 ||
				particleVertices[i].y <= 0.0f || particleVertices[i].y >= latticeHeight - 1 ||
				particleVertices[i].z <= 0.0f || particleVertices[i].z >= latticeDepth - 1) {

				particleVertices[i] = glm::vec3(0.0f, respawnY, respawnZ++);
				if (respawnZ >= latticeDepth - 1) {
					respawnZ = 0;
					respawnY++;
				}
				if (respawnY >= latticeHeight - 1) {
					respawnY = 0;
				}
			}
		}

	}
}

void LBM3D_1D_indices::updateInlets() {

	float weightMiddle = 1.0f / 3.0f;
	float weightAxis = 1.0f / 18.0f;
	float weightNonaxial = 1.0f / 36.0f;


	float macroDensity = 1.0f;
	glm::vec3 macroVelocity = inletVelocity;


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

	for (int z = 0; z < latticeDepth; z++) {
		for (int y = 0; y < latticeHeight; y++) {

			int idx = getIdx(0, y, z);

			backLattice[idx].adj[DIR_MIDDLE_VERTEX] = middleEq;
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


	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);

				if ((heightMap->data[x][z] >= y && heightMap->data[x][z] > 0.01f) || y == 0) {



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
				}
			}
		}
	}
}

void LBM3D_1D_indices::resetSimulation() {
	cout << "Resetting simulation..." << endl;
	particleSystem->initParticlePositions(latticeWidth, latticeHeight, latticeDepth, heightMap);
	for (int i = 0; i < latticeWidth * latticeHeight; i++) {
		for (int j = 0; j < 19; j++) {
			backLattice[i].adj[j] = 0.0f;
		}
		velocities[i] = glm::vec3(0.0f);
	}
	initLattice();


	cudaMemcpy(d_frontLattice, frontLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_backLattice, backLattice, sizeof(Node3D) * latticeSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocities, velocities, sizeof(glm::vec3) * latticeSize, cudaMemcpyHostToDevice);

}

void LBM3D_1D_indices::updateControlProperty(eLBMControlProperty controlProperty) {
	switch (controlProperty) {
		case MIRROR_SIDES_PROP:
			cudaMemcpyToSymbol(d_mirrorSides, &mirrorSides, sizeof(int));
			break;
	}
}


void LBM3D_1D_indices::switchToCPU() {
}

void LBM3D_1D_indices::initBuffers() {

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
	for (int x = 0; x < latticeWidth; x++) {
		for (int y = 0; y < latticeHeight; y++) {
			for (int z = 0; z < latticeDepth; z++) {
				int idx = getIdx(x, y, z);
				frontLattice[idx].adj[DIR_MIDDLE_VERTEX] = weightMiddle;
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


void LBM3D_1D_indices::swapLattices() {
	// CPU
	Node3D *tmp = frontLattice;
	frontLattice = backLattice;
	backLattice = tmp;

	// GPU
	tmp = d_frontLattice;
	d_frontLattice = d_backLattice;
	d_backLattice = tmp;
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
