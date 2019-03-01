///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       LBM.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines the abstract LBM class.
*
*  Defines the abstract LBM class. It has LBM2D and LBM3D children that simulate fluids in 2D and 3D.
*  It is used so that we can change simulation type using configuration file without recompiling the solution.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ShaderProgram.h"
#include "ParticleSystem.h"


/// The superclass of LBM simulators for this application.
/**
	LBM is the superclass of LBM simulators for this application.
	It defines the particular steps (pure virtual functions) of the simulation and its variables.
	It provides some basic functionality that is shared by simulators in 2D and 3D.
*/
class LBM {
public:

	/// Enumeration that should contain all controllable properties of the simulation (through UI).
	enum eLBMControlProperty {
		MIRROR_SIDES_PROP
	};

	ParticleSystem *particleSystem;		///< Pointer to the particle system
	glm::vec3 *particleVertices;		///< Pointer to the particle vertices array (on CPU)
	int *d_numParticles;	///< Number of particles on the device; managed in memory by Particle System class (its destructor)


	int latticeWidth;		///< Width of the lattice
	int latticeHeight;		///< Height of the lattice
	int latticeDepth;		///< Depth of the lattice

	int latticeSize;		///< Number of lattice nodes = latticeWidth * latticeHeight * latticeDepth

	float tau = 0.52f;		///< Tau parameter of the simulation, describes the viscosity of the simulated fluid
	float itau;				///< Inverse value of tau = 1.0 / tau; it is used in the collision step, if tau is changed, this value must be recomputed
	float nu;				///< Experimental value for subgrid model simulation

	glm::vec3 inletVelocity = glm::vec3(1.0f, 0.0f, 0.0f);		///< Inlet velocity vector

	int useSubgridModel = 0;	///< Whether the subgrid model should be used - EXPERIMENTAL - subgrid model not functional at the moment!
	int mirrorSides = 1;		///< Whether the particles passing through sides of the scene bounding box should show up on the other side		
	int visualizeVelocity = 0;  ///< Whether the velocity of the particles should be visualized, currently only in 2D
	int respawnLinearly = 0;	///< NOT USED YET! Whether particles should respawn linearly or randomly in the inlet nodes
	
	string sceneFilename;		///< Filename of the scene that is used for the simulation

	/// Default constructor.
	LBM();

	/// Constructs the LBM with specified scene and dimensions, and with initial tau value.
	/**
		Constructs the LBM with specified scene and dimensions, and with initial tau value.
		Inverse tau and nu values are computed from tau.
		\param[in] dimensions		Dimensions of the scene: vec3(latticeWidth, latticeHeight, latticeDepth).
		\param[in] sceneFilename	Filename of the scene.
		\param[in] tau				Initial value of tau.
		\param[in] particleSystem	Particle system that will be used.
	*/
	LBM(glm::ivec3 dimensions, string sceneFilename, float tau, ParticleSystem *particleSystem);

	/// Default virtual destructor.
	virtual ~LBM();

	/// Initializes the scene.
	virtual void initScene() = 0;

	/// Draws the scene.
	virtual void draw(ShaderProgram &shader) = 0;

	/// Does one step of the simulation.
	/**
		Does one step of the simulation.
		This step can be decomposed into multiple small steps:
			clearBackLattice, updateInlets, streamingStep, updateColliders, collisionStep, moveParticles, swap
	*/
	virtual void doStep() = 0;

	/// Does one step of the simulation using GPU device with CUDA capability.
	/**
		Does one step of the simulation using GPU device with CUDA capability.
		This step can be decomposed into multiple small steps (combination of kernels and regular functions):
		clearBackLattice, updateInlets, streamingStep, updateColliders, collisionStep, moveParticles, swap
	*/
	virtual void doStepCUDA() = 0;

	/// Clears the backLattice.
	virtual void clearBackLattice() = 0;

	/// Streaming step updates the distribution function in the back lattice.
	/**
		 Streaming step updates the distribution function in the back lattice 
		 based on the distribution function in the previous frame (frontLattice).
	*/
	virtual void streamingStep() = 0;

	/// Collision step computes micro collisions between particles.
	/**
		Collision step computes micro collisions between particles using Bhatnagar-Gross-Krook operator for finding equilibrium.
		The macroscopic velocity and density are first computed.
	*/
	virtual void collisionStep() = 0;

	/// Moves particles in the direction of the velocity of adjacent lattice nodes.
	/**
		Moves particles in the direction of the velocity of adjacent lattice nodes using interpolation.
	*/
	virtual void moveParticles() = 0;

	/// Updates the inlets of the scene to the set inlet velocity.
	/**
		Updates the inlets of the scene to the set inlet velocity.
		Macroscopic density is therefore set to 1.0 and the same collision operation is done as in the collision step.
	*/
	virtual void updateInlets() = 0;

	/// Updates lattice nodes where there are colliders/obstacles.
	/**
		Updates lattice nodes where there are colliders/obstacles by reversing their distribution function directions.
	*/
	virtual void updateColliders() = 0;

	/// Resets the simulation to intial state.
	virtual void resetSimulation() = 0;

	/// Updates the specified control property of the simulation.
	/**
		Updates the specified control property of the simulation.
		\param[in] controlProperty		The property to be updated.
	*/
	virtual void updateControlProperty(eLBMControlProperty controlProperty) = 0;

	/// Recalculates main simulation variables. Useful when the value of tau has been modified.
	virtual void recalculateVariables();

	/// Switches the simulation to CPU. (EXPERIMENTAL)
	virtual void switchToCPU() = 0;

protected:

	vector<glm::vec3> velocityArrows;	///< Array describing velocity arrows (starts in node, points in velocity direction) for visualization
	vector<glm::vec3> particleArrows;	///< Array describing velocity arrows that visualize particle velocity (interpolated values)

	GLuint velocityVBO;		///< VBO for node velocity visualization
	GLuint velocityVAO;		///< VAO for node velocity visualization

	GLuint particleArrowsVAO;	///< VAO for particle velocity (arrow) visualization
	GLuint particleArrowsVBO;	///< VBO for particle velocity (arrow) visualization

	/// Swaps the lattice pointers (front and back) on CPU and GPU.
	virtual void swapLattices() = 0;

	/// Initializes OpenGL buffers for drawing lattice nodes and velocity arrows.
	virtual void initBuffers() = 0;

	/// Initializes the lattice with initial distribution weights.
	virtual void initLattice() = 0;

};

