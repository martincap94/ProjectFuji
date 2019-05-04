///////////////////////////////////////////////////////////////////////////////////////////////////
/*!
* \file       LBM2D_1D_indices.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines the LBM2D class and data structures it uses (Node, directionVectors, eDirection).
*	
*	--- DEPRECATED ---
*	\deprecated
*	Defines the LBM2D class (subclass of LBM) that implements the 2D simulation using CPU and GPU.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Config.h"

#include "ShaderProgram.h"
#include "ParticleSystemLBM.h"
#include "LatticeCollider.h"

#include <cuda_gl_interop.h>


#include <vector>

#include "LBM.h"

// temporary -> will be moved to special header file to be shared
// among all classes (Node -> Node2D and Node3D)
// this applies to Node, vRight, ..., eDirection
/// Lattice node for 2D simulation (9 streaming directions -> 9 floats in distribution function).
struct Node {
	float adj[9];	///< Distribution function for adjacent nodes (in streaming directions).
};

// Streaming directions vectors
const glm::vec3 vRight = glm::vec3(1.0f, 0.0f, 0.0f);
const glm::vec3 vTop = glm::vec3(0.0f, 1.0f, 0.0f);
const glm::vec3 vLeft = glm::vec3(-1.0f, 0.0f, 0.0f);
const glm::vec3 vBottom = glm::vec3(0.0f, -1.0f, 0.0f);
const glm::vec3 vTopRight = glm::vec3(1.0f, 1.0f, 0.0f);
const glm::vec3 vTopLeft = glm::vec3(-1.0f, 1.0f, 0.0f);
const glm::vec3 vBottomLeft = glm::vec3(-1.0f, -1.0f, 0.0f);
const glm::vec3 vBottomRight = glm::vec3(1.0f, -1.0f, 0.0f);

/// Streaming directions array.
const glm::vec3 directionVectors[9] = {
	glm::vec3(0.0f, 0.0f, 0.0f),
	glm::vec3(1.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, 0.0f, 0.0f),
	glm::vec3(0.0f, -1.0f, 0.0f),
	glm::vec3(1.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, 1.0f, 0.0f),
	glm::vec3(-1.0f, -1.0f, 0.0f),
	glm::vec3(1.0f, -1.0f, 0.0f)
};


/// Streaming direction enum for 2D.
enum eDirection {
	DIR_MIDDLE = 0,
	DIR_RIGHT,
	DIR_TOP,
	DIR_LEFT,
	DIR_BOTTOM,
	DIR_TOP_RIGHT,
	DIR_TOP_LEFT,
	DIR_BOTTOM_LEFT,
	DIR_BOTTOM_RIGHT,
	NUM_2D_DIRECTIONS
};

/// 2D LBM simulator.
/**
	2D LBM simulator that supports both CPU and GPU simulations.
	GPU (CUDA) simulation is run through global kernels that are defined in LBM2D_1D_indices.cu.
	The LBM is indexed as a 1D array in this implementation.
	The simulator supports particle velocity visualization. Stream line and velocity vector
	visualizations have been deprecated.
*/
class LBM2D_1D_indices : public LBM {

	const float WEIGHT_MIDDLE = 4.0f / 9.0f;		///< Initial weight for the middle value in distribution function
	const float WEIGHT_AXIS = 1.0f / 9.0f;			///< Initial weight for all values in distribution function that lie on the axes
	const float WEIGHT_DIAGONAL = 1.0f / 36.0f;		///< Initial weight for all values in distribution function that lie on the diagonal


public:

	Node *frontLattice;			///< Front lattice - the one currently drawn at end of each frame
	Node *backLattice;			///< Back lattice - the one to which we prepare next frame to be drawn

	Node *d_frontLattice;		///< Device pointer for the front lattice
	Node *d_backLattice;		///< Device pointer for the back lattice
	glm::vec2 *d_velocities;	///< Device pointer to the velocities array

	bool *d_tCol;				///< Device pointer to the scene collider (scene descriptor)


	LatticeCollider *tCol;				///< Scene collider (scene descriptor)

	struct cudaGraphicsResource *cudaParticleVerticesVBO;	///< Device pointer that is mapped to particle vertices VBO
	struct cudaGraphicsResource *cudaParticleColorsVBO;		///< Device pointer that is mapped to particle colors VBO


	glm::vec2 *velocities;				///< Macroscopic velocities array


	/// Default constructor.
	LBM2D_1D_indices();

	/// Constructs the simulator with given dimensions, scene, initial tau value and number of threads for launching kernels.
	/**
		Constructs the simulator with given dimensions, scene, initial tau value and number of threads for launching kernels.
		Initializes the scene and allocates CPU and GPU memory for simulation.
		\param[in] dim				[NOT USED ANYMORE] Dimensions of the scene. Dimensions are now loaded from the scene file.
		\param[in] sceneFilename	Filename of the scene. Scene defines the dimensions of the simulation space.
		\param[in] tau				Initial tau simulation constant.
		\param[in] particleSystem	Pointer to the particle system.
		\param[in] numThreads		Number of threads per block to be used when launching CUDA kernels.
	*/
	LBM2D_1D_indices(glm::ivec3 dim, string sceneFilename, float tau, ParticleSystemLBM *particleSystem, int numThreads);

	/// Frees CPU and GPU memory and unmaps CUDA graphics resources (VBOs).
	virtual ~LBM2D_1D_indices();


	// All the functions below inherit its doxygen documentation from the base class LBM (with some exceptions).

	virtual void recalculateVariables();

	virtual void initScene();

	virtual void draw(ShaderProgram &shader);

	virtual void doStep();
	virtual void doStepCUDA();

	virtual void clearBackLattice();
	virtual void streamingStep();
	virtual void collisionStep();

	/// Streamlined version of the collision step.
	/**
		Does the collision step of the simulation without using that many temporary variables and dot products (hence streamlined).
		Was tested as possible speedup (both on CPU and GPU), unfortunately, this version is a little bit slower
		than its original/basic counterpart.
	*/
	void collisionStepStreamlined();

	virtual void moveParticles();
	virtual void updateInlets();
	virtual void updateColliders();

	virtual void resetSimulation();


	/// Copies data from GPU memory back to CPU memory.
	/**
		Copies data from GPU memory back to CPU memory. This includes copying the particle vertices VBO back to host/CPU side.
		This is useful when we want to switch from CUDA to CPU at runtime without changing the state of the simulation.
	*/
	virtual void switchToCPU();
	virtual void synchronize();

protected:

	virtual void swapLattices();
	virtual void initBuffers();
	virtual void initLattice();

private:

	int numThreads;			///< Number of threads per block for kernel launches
	int numBlocks;			///< Number of blocks in grid for kernel launches

	GLuint vbo;				///< VBO for lattice node points
	GLuint vao;				///< VAO for lattice node points
	

	int respawnIndex = 0;		///< Respawn index (y coordinate) for the simulation
	int respawnMinY;			///< Minimum y respawn coordinate (when the inlet is blocked by an obstacle)
	int respawnMaxY;			///< Maximum y respawn coordinate (when the inlet is blocked by an obstacle)

	int streamLineCounter = 0;	///< Counter of streamlines for draw calls

	/// Precomputes respawnMinY and respawnMaxY (range, where the particles can respawn).
	void precomputeRespawnRange();

	/// Returns flattened index.
	int getIdx(int x, int y) {
		return x + y * latticeWidth;
	}

	/// Calculates macroscopic density for the lattice node.
	float calculateMacroscopicDensity(int x, int y);

	/// Calculates macroscopic velocity for the lattice ndoe with a given macroscopic density.
	glm::vec3 calculateMacroscopicVelocity(int x, int y, float macroDensity);



};

